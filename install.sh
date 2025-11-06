git submodule update --init --recursive
cd 3rdparty/Megatron-LM && git remote add upstream https://github.com/NVIDIA/Megatron-LM.git && git fetch upstream && git checkout upstream/dev
# pip install torch -U --index-url https://download.pytorch.org/whl/cu129
# pip install --editable .
# pip install --no-build-isolation "transformer_engine[pytorch] @ git+https://github.com/nvidia/transformerengine.git"
