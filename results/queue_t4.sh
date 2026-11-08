#!/bin/zsh
# T4 queue, 2026-11-01 (second VM): the frame fc grid (finished fits in results/frame_fc are uploaded and skipped), then
# the frame streaming fit with the background mode and fc learned, short batch revisits, 8 epochs.
cd ~/code_repo/blml
SRC=/private/tmp/claude-501/-Users-ra/e6cd45ce-b6f4-42cc-be9a-45c946facadf/scratchpad/gpu_data
rm -f results/.frame_fc.session
BLML_DATA_SRC=$SRC BLML_FC_MULTS=0.8,0.9,1,1.1,1.22,1.35 ./colab_job.sh start frame_fc frame_fc_gpu_job.py T4
[ -f results/.frame_fc.session ] && { colab stop -s "$(cat results/.frame_fc.session)" || true; rm -f results/.frame_fc.session; }
rm -f results/.frame_bg.session
BLML_DATA_SRC=$SRC BLML_PARTS=stream BLML_EPOCHS=8 BLML_REVISIT_SWEEPS=300 ./colab_job.sh start frame_bg frame_bg_gpu_job.py T4
