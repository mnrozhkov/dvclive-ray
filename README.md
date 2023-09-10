To setup a Ray cluster, run:

```console
$ ray up tune-default.yaml
```

To run the script on the cluster:

```console
$ ray submit tune-default.yaml hf.py -- --address=localhost:6379
```
