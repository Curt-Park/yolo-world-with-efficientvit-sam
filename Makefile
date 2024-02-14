YOLO_WORLD_URL := "https://huggingface.co/wondervictor/YOLO-World/resolve/main"
YOLO_WORLD_MODEL := "yolo_world_l_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained-0e566235.pth"
EFFICIENT_SAM_URL := "https://huggingface.co/han-cai/efficientvit-sam/resolve/main"
EFFICIENT_SAM_MODEL := "xl1.pt"


define download
	@if [ ! -f $(2) ]; then \
		echo "Download $(2)..."; \
		wget "$(1)/$(2)"; \
	fi
endef


setup:
	pip install -r requirements.txt
	mim install 'mmengine==0.10.3'
	mim install 'mmcv-lite==2.0.0'
	mim install 'mmdet==3.3.0'
	mim install 'mmyolo==0.6.0'


model:
	$(call download,$(YOLO_WORLD_URL),$(YOLO_WORLD_MODEL))
	$(call download,$(EFFICIENT_SAM_URL),$(EFFICIENT_SAM_MODEL))
