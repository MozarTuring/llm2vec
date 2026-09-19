export PKQ_ARCH="aarch64"
export PKQ_MODULES="GPU/Miniforge/26.3.2-2-eb GPU/buildtool-easybuild/5.2.1-hpca3ef7d197 CUDA/12.9.1 cuDNN/9.15.0.57-CUDA-12.9.1 cuSPARSELt/0.8.0.4-CUDA-12.9.1"
if [ -z  ]; then
    export PKQ_CONDAENV=/nobackup/proj/disk/naiss2026-3-658/personal/pkquser/pkqcondaenv/llm2vec_pikaq
    export PKQ_WHEELS=/nobackup/proj/disk/naiss2026-3-658/personal/pkquser/pkqwheels/llm2vec_pikaq
fi
echo "condaenv path "
