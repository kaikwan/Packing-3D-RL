# Packing-3D-RL
Fork of yang-d19/Packing-3D-RL

### 1. Overview

This is a pip package that provides an implementation to ***Stable bin packing of non-convex 3D objects with a robot manipulator, Fan Wang and Kris Hauser.***  *(arXiv:1812.04093v1 [cs.RO] 10 Dec 2018)*. In PRL, this is used with GCULab to measure container fullness.

### 2. Installation
```
git clone https://github.com/kaikwan/Packing-3D-RL.git
cd Packing-3D-RL
pip install -e .
```

### 3. Demonstration
```
python demo.py
```

### 3. Functions of each file

Below are explanations to some files:

1. `demo.py` provides a demo that use heuristic method to solve bin packing problem. If it runs successfully,  you will see the window as below. 

   <img src="pictures\demo.png" alt="demo" style="zoom:20%;" />
2. `utils.py` provides basic classes that many other files may use, including `Position`, `Attitude`, `Transform` and other classes concerning the status of a single object.

3. `object.py` provides definition of class `Geometry`  and `Item`, `Geometry` mainly contains geometric transformation and heightmap calculation. `Item` wraps up `Geometry` , makes it more accessible to upper classes.

4. `container.py` defines class `Container`, it provides manipulations of searching possible positions and adding an item to the container.

5. `pack.py` is the top-level class, it defines how to solve a bin packing problem, it provides functions about loading a sequence of objects, automatically adding them to a container, and clearing the container.

6. `show.py` defines class `Display`, it visualizes the procedure of bin packing decision. You only need to pass an object of type `Geometry` to its function `show3d()`, then it will show you how this geometry looks like.


### 4. Contributors

#### Original Author

Yang Ding

Department of Automation, Tsinghua University

Email: yangding19thu@163.com

Web: https://yang-d19.github.io/
