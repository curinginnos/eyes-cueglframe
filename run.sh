cd /usr/src/jetson_multimedia_api/argus/build/samples/syncSensor

./argus_syncsensor --duration 10


# sudo systemctl restart nvargus-daemon.service

# nvprof ./argus_syncsensor --duration 10
# cuda-memcheck ./argus_syncsensor --duration 10
# ./argus_syncsensor --duration 10