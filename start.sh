#!/usr/bin/env bash
###### add volume #####
echo "Worker Initiated"
echo "Symlinking files from Network Volume"
rm -rf /workspace && \
  ln -s /runpod-volume /workspace

#######
# Use libtcmalloc for better memory management
if [ -f "/workspace/venv/bin/activate" ]; then
    echo "Starting WebUI API"
    source /workspace/venv/bin/activate
    TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
    export LD_PRELOAD="${TCMALLOC}"

    # Serve the API and don't shutdown the container
    if [ "$SERVE_API_LOCALLY" == "true" ]; then
        echo "runpod-worker-comfy: Starting ComfyUI"
        python3 /workspace/comfyui/main.py --disable-auto-launch --disable-metadata --listen &

        echo "runpod-worker-comfy: Starting RunPod Handler"
        python3 -u /rp_handler.py --rp_serve_api --rp_api_host=0.0.0.0
    else
        echo "runpod-worker-comfy: Starting ComfyUI"
        python3 /workspace/comfyui/main.py --disable-auto-launch --disable-metadata &

        echo "runpod-worker-comfy: Starting RunPod Handler"
        python3 -u /rp_handler.py
    fi
      deactivate
else
    echo "ERROR: The Python Virtual Environment (/workspace/venv/bin/activate) could not be activated"
    echo "Ensure that you have attached your Network Volume to your endpoint."
fi