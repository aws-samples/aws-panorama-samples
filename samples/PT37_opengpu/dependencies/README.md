# Build instructions

```
podman build -v /usr/bin/qemu-aarch64-static:/usr/bin/qemu-aarch64-static -t "panoramasdk":r32.4.3 \
	--build-arg "RELEASE=r32.4" --build-arg "CUDA=10.2" \
	-f ./Dockerfile ./
```

# QEMU installation for building on x86 

```
sudo apt-get install qemu binfmt-support qemu-user-static
```

# libpod Installation Instructions


#### [Ubuntu](https://www.ubuntu.com)

```bash
sudo apt-get update -qq
sudo apt-get install -qq -y software-properties-common uidmap
sudo add-apt-repository -y ppa:projectatomic/ppa
sudo apt-get update -qq
sudo apt-get -qq -y install podman
```


