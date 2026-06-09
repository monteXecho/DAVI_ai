# DAVI AI Module

## Prerequisites

For convenience and shared ownership, we run all as root.
Start with:
```
sudo su
```

Install `uv` | [Installing uv](https://docs.astral.sh/uv/getting-started/installation/)

E.g. (given linux distro):
```bash
cd /
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install `vllm` | [vLLM installation](https://docs.vllm.ai/en/latest/getting_started/installation/)

```bash
mkdir vllm-deployment
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
```

Note: for both (esp `uv`), can be run from any directory. Here we chose to run first installation from root home, and created a dedicated folder for vllm-deployment.

## Running Everything

### [Optional] Running in tmux

For convenience and be able to detach sessions, we can run all the services in `tmux`
```
tmux
```


### Running the VLLM server

In the first tmux window (default), we run vllm.

Run with the following config:
```
cd vllm-deployment
uv run vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 4400 --max-model-len 12288 --gpu-memory-utilization 0.7
```

_Optional_: In a new tmux window (`Ctrl-B + C`), watch the GPU usage with:
```
watch nvidia-smi
```

### Running the AI app

_Optional_: If using tmux, can create a new window for this. `Ctrl-B + C` to create a new one.

`cd` into the DAVI_ai module (wherever it's been cloned).

`cp setup_env_sample.sh setup_env.sh` to fill in this env specific variables.

Fill in the needed variables:

```
export HAYHOOKS_PIPELINES_DIR="$(pwd)/hayhooks_wrapper"
export HAYHOOKS_ADDITIONAL_PYTHONPATH="$(pwd)"
export PYTHONPATH="$(pwd):$PYTHONPATH"
# as we switch index per request, can set any default
# say: "davi-base"
export INDEX_NAME="<fill>" 
export DOC_STORE_TYPE=OpenSearch
export OPENSEARCH_HOST="<fill>" # Likely: "https://werkh-2.vm.bit.nl:9200"
export OPENSEARCH_USE_SSL=TRUE
export OPENSEARCH_CA_CERTS=<fill> # Likely: /path/to/http-ca.pem
# See: Opensearch-Setup
export OPENSEARCH_USERNAME=<fill> # See: Opensearch-Setup or ask admin
export OPENSEARCH_PASSWORD=<fill> # See: Opensearch-Setup or ask admin
export OPENSEARCH_EMBEDDING_DIM=768
export OPENAI_API_KEY="..." # Empty is fine
export LLM_BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
export LLM_BASE_URL="http://0.0.0.0:4400"
export SERPERDEV_API_KEY="<fill>" # Ask admin
export HF_TOKEN="<fill>" # Ask admin
```

Export/setup these env variables
```
source setup_env.sh
```

Finally, run hayhooks server with:
```
./run_hayhooks.sh
```

### iptables Configuration

If needed (probably is needed by default), the iptables can be configured as below to allow access to the app from docker containers

```
sudo iptables -I INPUT -i docker+ -p tcp --dport 1416 -j ACCEPT
sudo iptables -I INPUT -i br-+ -p tcp --dport 1416 -j ACCEPT
```

And to verify:
```
sudo iptables -L INPUT -n -v
```

## Set up on the OpenSearch server

The following is only necessary if the admin(s) have not provided you with the cert file and the (username, password), that's required to connect to the opensearch server from the main server.

All / most of the below is on the opensearch server itself.
We're assuming you have root access. If not you'd have to ask the bit-nl representative / admin.

First assumption: the opensearch server has been set up with TLS enabled (but not mTLS).
To check, in `/etc/opensearch/opensearch.yml` we see:
 `plugins.security.ssl.http.enabled: true` but not `plugins.security.ssl.http.clientauth: REQUIRE` 

### Setting up a new user credentials

If we know the admin password, we can create the users as:

```bash
curl -k -u admin:ADMIN_PASSWORD \
  -X PUT https://localhost:9200/_plugins/_security/api/internalusers/my_app \
  -H "Content-Type: application/json" \
  -d '{
    "password": "<your-chosen-password>"
  }'
curl -k -u admin:ADMIN_PASSWORD \
  -X PUT https://localhost:9200/_plugins/_security/api/rolesmapping/my_app_role \
  -H "Content-Type: application/json" \
  -d '{
    "users": ["my_app"]
  }'
```

If not, we use some available scripts. 
This needs `java` . 
If it’s not installed (check with `java -version` ), install java:

```bash
sudo apt-get update
sudo apt-get install -y default-jre
```

Then retrieve the security config into a new folder and overwrite: 

```bash
sudo mkdir -p /tmp/os-security-backup
sudo cd /usr/share/opensearch/plugins/opensearch-security/tools/
sudo ./securityadmin.sh \
  -cacert /etc/opensearch/root-ca.pem \
  -cert /etc/opensearch/kirk.pem \
  -key /etc/opensearch/kirk-key.pem \
  -r \
  -cd /tmp/os-security-backup \
  -nhnv
```

Here, the files likely have a timestamp associated with them such as `2026-Jan-18_07-00-00` , we rename and copy into a new folder:

Create and run a convenience script: `copy_config.sh` as below:

```bash
ts="2026-Jan-18_07-00-00" # Replace with what you have

sudo cp "/tmp/os-security-backup/action_groups_${ts}.yml"      /tmp/os-security-working/action_groups.yml
sudo cp "/tmp/os-security-backup/audit_${ts}.yml"              /tmp/os-security-working/audit.yml
sudo cp "/tmp/os-security-backup/allowlist_${ts}.yml"          /tmp/os-security-working/allowlist.yml
sudo cp "/tmp/os-security-backup/config_${ts}.yml"             /tmp/os-security-working/config.yml
sudo cp "/tmp/os-security-backup/internal_users_${ts}.yml"     /tmp/os-security-working/internal_users.yml
sudo cp "/tmp/os-security-backup/nodes_dn_${ts}.yml"           /tmp/os-security-working/nodes_dn.yml
sudo cp "/tmp/os-security-backup/roles_${ts}.yml"              /tmp/os-security-working/roles.yml
sudo cp "/tmp/os-security-backup/roles_mapping_${ts}.yml"      /tmp/os-security-working/roles_mapping.yml
sudo cp "/tmp/os-security-backup/security_tenants_${ts}.yml"   /tmp/os-security-working/tenants.yml

```

Now, create a password hash (using the default available tool):

```bash
cd /usr/share/opensearch/plugins/opensearch-security/tools/
./hash.sh -p '<your-chosen-password>'
```

Update users file:

```bash
sudo vi /tmp/os-security-working/internal_users.yml
```

with:

```bash
davi_app:
  hash: "$2y$12$PASTE_HASH_HERE"
  reserved: false
  backend_roles: []
  description: "Application user"
```

Map user to a role:

```bash
sudo vi /tmp/os-security-working/roles_mapping.yml
```

```bash
all_access:
  users:
    - "davi_app"
```

Upload the config from the working directory:

```bash
cd /usr/share/opensearch/plugins/opensearch-security/tools/

sudo ./securityadmin.sh \
  -cacert /etc/opensearch/root-ca.pem \
  -cert /etc/opensearch/kirk.pem \
  -key /etc/opensearch/kirk-key.pem \
  -cd /tmp/os-security-working/ \
  -nhnv

```

Test:

```bash
curl --cacert /etc/opensearch/root-ca.pem -u davi_app:<your-chosen-password> https://localhost:9200/_cat/plugins?v
```

### Setting up a CA and server certificate

Apart from the username-password, the client also needs a tls certificate.

Create a new CA:

```bash
sudo mkdir -p /etc/opensearch/pki
cd /etc/opensearch/pki

sudo openssl genrsa -out http-ca.key 4096
sudo openssl req -x509 -new -nodes -key http-ca.key -sha256 -days 3650 \
  -subj "/C=DE/O=opensearch/OU=http/CN=opensearch-http-ca" \
  -out http-ca.pem
```

Create a new HTTP server cert for `werkh-2.vm.bit.nl` 

```bash
sudo openssl genrsa -out werkh-2-http.key 2048
sudo openssl req -new -key werkh-2-http.key \
  -subj "/C=DE/O=node/OU=node/CN=werkh-2.vm.bit.nl" \
  -out werkh-2-http.csr

cat > werkh-2-http.ext <<'EOF'
subjectAltName = @alt_names
extendedKeyUsage = serverAuth

[alt_names]
DNS.1 = werkh-2.vm.bit.nl
DNS.2 = werkh-2
EOF

sudo openssl x509 -req -in werkh-2-http.csr \
  -CA http-ca.pem -CAkey http-ca.key -CAcreateserial \
  -out werkh-2-http.pem -days 825 -sha256 \
  -extfile werkh-2-http.ext
```

Note: set to expire in 825 days.

Point OpenSearch HTTP TLS to the new cert/key/CA and restart

Edit `opensearch.yml` (paths depend on your install, likely `/etc/opensearch/opensearch.yml`; below assumes you put files in /etc/opensearch/pki):

```bash
plugins.security.ssl.http.enabled: true
plugins.security.ssl.http.pemcert_filepath: /etc/opensearch/pki/werkh-2-http.pem
plugins.security.ssl.http.pemkey_filepath: /etc/opensearch/pki/werkh-2-http.key
plugins.security.ssl.http.pemtrustedcas_filepath: /etc/opensearch/pki/http-ca.pem
```

Also might need to change ownership and permissions:

```bash
sudo chown -R opensearch:opensearch /etc/opensearch/pki
sudo chmod 750 /etc/opensearch/pki
sudo chmod 640 /etc/opensearch/pki/*.pem
sudo chmod 640 /etc/opensearch/pki/*.key
```

Restart:

```bash
sudo systemctl restart opensearch
```

Copy the cert (`/etc/opensearch/pki/http-ca.pem`) to any connecting server (i.e. the one where we've the ai app). (e.g. through `scp` by first downloading from OpenSearch server and then uploading to the ai app VM/server).

Verify from the server:
```bash
curl --cacert /etc/opensearch/http-ca.pem -u davi_app:<your-chosen-password> https://werkh-2.vm.bit.nl:9200
```

So, in the `setup_env.sh`, we include:
```bash
...
export OPENSEARCH_HOST="https://werkh-2.vm.bit.nl:9200"
export OPENSEARCH_USE_SSL=TRUE
export OPENSEARCH_CA_CERTS="/path/where/you/copied/http-ca.pem"
export OPENSEARCH_USERNAME="davi_app"
export OPENSEARCH_PASSWORD="<your-chosen-password>"
export OPENAI_API_KEY="..." # Empty is fine
```
