import importlib
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union, get_args, get_origin

import tenacity
from haystack import component as component_decorator, default_from_dict, default_to_dict
from haystack.core.component import Component
from haystack.core.errors import PipelineError
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.logging import getLogger
from tqdm import tqdm

logger = getLogger(__name__)


def is_optional_type(_type: Type) -> bool:
    return get_origin(_type) is Union and type(None) in get_args(_type)


def listify_type(_type: Type, flatten: bool = False) -> Type:
    if flatten and get_origin(_type) is list:
        return _type
    if is_optional_type(_type):
        return Optional[List[_type]]  # type: ignore

    return List[_type]  # type: ignore


@component_decorator
class DeepsetParallelExecutor:
    """
    Runs another component in parallel with multiple inputs.

    This component takes a component and runs it in parallel with multiple inputs using a thread pool executor. The
    inputs are passed as lists of the component inputs. This is useful when you have a component that is slow and you
    want to run it in parallel with multiple inputs: for example running multiple llm invocations.
    """

    def __init__(
        self,
        component: Component,
        max_workers: int = 4,
        max_retries: int = 3,
        progress_bar: bool = False,
        raise_on_failure: bool = True,
        flatten_output: bool = False,
    ) -> None:
        """
        Creates an instance of DeepsetParallelExecutor.

        :param component: The component to run in parallel.
        :param max_workers: The maximum number of workers to use in the thread pool executor.
        :param max_retries: The maximum number of retries to attempt if the component fails.
        :param progress_bar: Whether to show a progress bar while running the component in parallel.
        :param raise_on_failure: Whether to raise an exception if the component fails.
        :param flatten_output: Whether to flatten the output of the component.
        """
        self._max_workers = max_workers
        self._max_retries = max_retries
        self._component = component
        self._progress_bar = progress_bar
        self._raise_on_failure = raise_on_failure
        self._flatten_output = flatten_output

        for name, socket in component.__haystack_input__._sockets_dict.items():  # type: ignore[attr-defined]
            listified_type = listify_type(socket.type)
            default_value = socket.default_value
            component_decorator.set_input_type(instance=self, name=name, type=listified_type, default=default_value)

        component_decorator.set_output_types(
            self,
            **{
                name: listify_type(socket.type, flatten=self._flatten_output)
                for name, socket in component.__haystack_output__._sockets_dict.items()  # type: ignore[attr-defined]
            },
        )

    def run(self, **kwargs: Iterable[Any]) -> Dict[str, List[str | Dict[str, Any]]]:
        """
        Runs the component in parallel with multiple inputs.

        :param kwargs: The inputs to the component. Each input must be a list of component inputs.
        :returns: The outputs of the component. Each output is a list.
        """
        logger.info(
            f"Running component {self._component.__class__.__name__} in parallel using {self._max_workers} workers"
        )
        # ignore default values
        kwargs = {name: values for name, values in kwargs.items() if isinstance(values, Iterable)}

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            # convert dict of lists into list of dicts
            kwargs_list = (dict(zip(kwargs, values, strict=True)) for values in zip(*kwargs.values(), strict=True))
            futures: List[Future] = []
            for kwargs_dict in kwargs_list:
                future = executor.submit(self._run_on_thread, **kwargs_dict)
                futures.append(future)

            for num, future in tqdm(
                enumerate(as_completed(futures), start=1),
                total=len(futures),
                disable=not self._progress_bar,
                desc=f"Running component {self._component.__class__.__name__} in parallel",
            ):
                _, statistics = future.result()
                if not self._progress_bar:
                    logger.info(
                        f"Running component {self._component.__class__.__name__} in parallel: {num} / "
                        f"{len(futures)} done",
                        extra={"execution_statistics": statistics},
                    )

            # results are ordered based on the order of the inputs
            results: List[Dict[str, Any]] = [future.result()[0] for future in futures]

            # convert list of dicts into dict of lists
            non_none_results = [result for result in results if result is not None]
            keys = non_none_results[0].keys() if non_none_results else []
            results_dict = {key: [d[key] if d is not None else None for d in results] for key in keys}

            if self._flatten_output:
                for key in keys:
                    values = results_dict[key]
                    if all(isinstance(v, list) for v in values):
                        results_dict[key] = [item for sublist in values for item in sublist]

        return results_dict

    def warm_up(self) -> None:
        """
        Warms up the component by running it once before the actual run.
        """
        if hasattr(self._component, "warm_up"):
            self._component.warm_up()

    def _run_on_thread(self, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        start_time = time.time()

        @tenacity.retry(stop=tenacity.stop_after_attempt(self._max_retries + 1), reraise=True)
        def run_component_with_retry() -> Any:
            try:
                return self._component.run(**kwargs)
            except Exception as e:
                logger.warning(
                    f"Component {self._component.__class__.__name__} failed with exception '{e}'.",
                    extra={"kwargs": kwargs},
                    exc_info=True,
                )
                raise

        result: Any = None
        try:
            result = run_component_with_retry()
        except Exception as e:
            msg = (
                f"Component {self._component.__class__.__name__} failed with exception '{e}' after "
                f"{self._max_retries} retries."
            )
            if self._raise_on_failure:
                logger.error(msg, extra={"kwargs": kwargs}, exc_info=True)
                raise
            logger.warning(msg, extra={"kwargs": kwargs}, exc_info=True)

        total_time = time.time() - start_time
        attempts = run_component_with_retry.statistics.get("attempt_number", 1)  # type: ignore
        statistics = {"total_time": total_time, "attempts": attempts, "success": result is not None}

        return result, statistics

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeepsetParallelExecutor":
        """
        Deserialize this component from a dictionary.

        :param data:
            The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        if component := data["init_parameters"].get("component"):
            if isinstance(component, dict):
                component_instance = cls._load_component(component)
                data["init_parameters"]["component"] = component_instance
        return default_from_dict(cls, data)  # type: ignore

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        return default_to_dict(  # type: ignore
            self,
            max_workers=self._max_workers,
            max_retries=self._max_retries,
            component=component_to_dict(self._component, "component"),
            progress_bar=self._progress_bar,
            raise_on_failure=self._raise_on_failure,
            flatten_output=self._flatten_output,
        )

    @classmethod
    def _load_component(cls, component_data: Dict[str, Any]) -> Component:
        if component_data["type"] not in component_decorator.registry:
            try:
                # Import the module first...
                module, _ = component_data["type"].rsplit(".", 1)
                logger.debug("Trying to import module {module_name}", module_name=module)
                importlib.import_module(module)
                # ...then try again
                if component_data["type"] not in component_decorator.registry:
                    raise PipelineError(
                        f"Successfully imported module {module} but can't find it in the component registry."
                        "This is unexpected and most likely a bug."
                    )
            except (ImportError, PipelineError) as e:
                raise PipelineError(f"Component '{component_data['type']}' not imported.") from e

        # Create a new one
        component_class = component_decorator.registry[component_data["type"]]
        instance = component_from_dict(component_class, component_data, "")
        return instance  # type: ignore[no-any-return]