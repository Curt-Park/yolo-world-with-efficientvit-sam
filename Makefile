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


model:
	$(call download,$(EFFICIENT_SAM_URL),$(EFFICIENT_SAM_MODEL))
