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

## Github repository
```bash
git clone https://github.com/United-Link/machine-translation.git
git checkout MultipleGPUs

cd machine-translation
docker build -t translator:mg .

```