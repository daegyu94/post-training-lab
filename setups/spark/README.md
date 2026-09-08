# NVIDIA DGX Spark Cluster Setup

이 저장소의 실행 환경은 controller 한 대와 NVIDIA DGX Spark 노드 `spark1`, `spark2`로 구성됩니다.
Controller는 Git checkout, 설정, 실행 기록을 관리하고 GPU 학습 process를 실행하지 않습니다.
각 Spark 노드는 자체 CUDA Python 환경과 node-local 모델 snapshot을 가지며 실제 rank process를 실행합니다.

```text
                           SSH control plane
 +--------------------+  ssh spark@spark1  +--------------------+
 | controller         | -----------------> | spark1             |
 |                    |  ssh spark@spark2  | rank 0 / GPU work  |
 | experiments/run.py | -----------------> +--------------------+
 | controller logs    |                              |
 +--------------------+                              | rendezvous + NCCL
          |                                          | (MASTER_ADDR and
          | NFS: /home/daegyu/shared                 |  configured interface)
          |                                          v
          |                              +--------------------+
          +----------------------------> | spark2             |
             NFS: /home/spark/shared     | rank 1 / GPU work  |
                                         +--------------------+
```

`experiments/run.py --execute`는 controller에서 각 노드로 SSH 접속해 해당 노드의 launcher를 시작합니다.
분산 학습 process는 `MASTER_ADDR`와 `MASTER_PORT`로 rendezvous한 뒤 NCCL이 setup에 지정된 통신 interface를 사용합니다.
물리 NIC, IP 대역, RoCE·InfiniBand 구성은 노드 환경마다 다르므로 `local.json`의 host, rendezvous 주소와 `NCCL_*` 변수를 실제 구성에 맞춰 설정합니다.

공유 저장소와 공유 데이터·출력은 controller에서 `/home/daegyu/shared`, Spark 노드에서 `/home/spark/shared`로 보입니다.
설정 파일의 `checkout`, `model_dirs`, `data_dir`, `output_root`에는 명령을 실행하는 노드에서 보이는 절대 경로를 적습니다.
