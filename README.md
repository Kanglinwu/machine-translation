# Machine Translation

## System Installation
```bash
sudo passwd ulink
sudo vim /etc/hostname
sudo vim /etc/hosts
```

```bash
sudo apt update
sudo apt upgrade
sudo apt install build-essential tmux vim sox ffmpeg htop nvtop git ca-certificates curl gnupg
sudo apt autoclean
sudo apt autoremove
```

## NVIDIA DRIVER Installation
```bash
sudo vim /etc/modprobe.d/blacklist-nouveau.conf

blacklist nouveau
options nouveau modeset=0

sudo update-initramfs -u
sudo reboot now
```
```bash
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo reboot now
```

## Docker
```bash
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
```
```bash
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker ulink
```
```bash
sudo systemctl enable docker
sudo systemctl enable containerd
sudo reboot now
```

## NVIDIA Container Toolkit
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/ubuntu22.04/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure
sudo systemctl restart docker
```

## Run
```bash
git clone https://github.com/United-Link/machine-translation.git

cd machine-translation
docker build -f Dockerfile_trans -t translator:v1.0.0 .
docker build -f Dockerfile_nginx -t nginx_load_balancer:v1.0.0 .
```
```bash
docker compose up -d
```

## Run by docker-swarm
```bash
# docker swarm init
docker swarm init

# run the service
docker stack deploy -c docker-compose_new.yml machine_translation

# delete the service
docker stack rm machine_translation

# check the service / log
docker service ls
docker service logs <service_name>
```

## API
### Post
```bash
curl -X POST http://10.10.10.119:2486/translate \
    -H "Content-Type: application/json" \
    -d '{"raw_text": "Cô gái này đẹp quá! ❤️❤️❤️❤️❤️ Anh yêu em!", "target_language": "English"}'
```
### Response
```bash
{
    "request_id": "25320973-3d3e-4b69-9728-b5cf67bec897", 
    "raw_text": "Cô gái này đẹp quá! ❤️❤️❤️❤️❤️ Anh yêu em!", 
    "translated_text": "This girl is beautiful! ❤️❤️❤️❤️❤️ I love you!", 
    "predicted_language": "Emoji", 
    "target_language": "English", 
    "model_lid": "facebook/fasttext-language-identification", 
    "model_mt": "facebook/nllb-200-distilled-1.3B"}
}
```


