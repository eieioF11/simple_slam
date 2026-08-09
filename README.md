# simple_slam

simple_slam

## Install Boost

```bash
sudo apt-get install libboost-all-dev
```

## Install iridescence

https://github.com/koide3/iridescence

```bash
sudo apt update && sudo apt install -y libiridescence-dev
```

## Install gtsam

```bash
# Add PPA
sudo add-apt-repository ppa:borglab/gtsam-release-4.0
sudo apt update  # not necessary since Bionic
# Install:
sudo apt install libgtsam-dev libgtsam-unstable-dev
```

## build

```bash
cd ~/ros2_ws/src
git clone https://github.com/eieioF11/simple_slam.git
cd ~/ros2_ws
colcon build --symlink-install
```

## launch

```bash
ros2 launch simple_slam simple_slam.launch.py
```
