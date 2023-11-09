# Test 

Run script 
```bash
python test/script.py
```


To setup a Ray cluster, run:

```console
ray up -y test/config.yaml
```

To run the script on the cluster:

```console
ray exec test/config.yaml 'python -c "import ray; ray.init()"'
```
