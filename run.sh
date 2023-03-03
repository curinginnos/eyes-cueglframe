cd /usr/src/jetson_multimedia_api/argus/build/samples/syncSensor

sudo systemctl restart nvargus-daemon.service

./argus_syncsensor --duration 30





# sudo systemctl restart nvargus-daemon.service

# nvprof ./argus_syncsensor --duration 10
# cuda-memcheck ./argus_syncsensor --duration 10
# ./argus_syncsensor --duration 10
