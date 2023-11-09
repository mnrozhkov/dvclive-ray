To setup a Ray cluster, run:

```bash
ray up tune-default.yaml  
```

To run the script on the cluster:

```bash
ray submit tune-default.yaml hf.py -- --address=localhost:6379
```


Synchronizing files from the cluster

```bash
ray rsync_up cluster.yaml 'dvclive' 'dvclive'
```

## Cluster/example-full.yaml

```bash
ray up cluster/example-full.yaml  

ray submit cluster/example-full.yaml hf.py -- --address=localhost:6379

```