# What it is
Applies sharpen (part 1) and gaussian blurr (filters) to the given image using cpu and cuda implementations

# How to run
#### compile 
    nvcc part1.cu -o part1
#### run 
    ./part1 image_grey.txt
    ./part2 image_color.txt

# Python Util code

### Show a txt into png
    python3 txt2imageXXX.py xxxx.txt
### Generate a RGB test set
    python3 image2txt.py xxxx.jpg/png

# Some design choices

Following are some design choices I wanted to declare for my solution

1. Flipping the filter: Given the way convolution is calculated, flipping the filter and then placing it on the input gives us a direct mapping for multiplication of values. 
2. Different implementation for padding: Instead of explicitly adding zeros for my padding, you will observe I do something different. The actual influence of padding is seen inside the `CPU_Test` and `GPU_Test` implementations in the `if` condition - `if (i > -1 && i < sizeY && j > -1 && j < sizeX)`. This implementation works as follows. Firstly, the mid point of the filter is actually placed on the current input index. We know that for any `(i, j)` in the input, there will be a `(i, j)` in the output calculated using the filter. After the midpoint of the filter is placed at `(i, j)`, we calculate at what position of the input (and output), the `(0, 0)` of the index lies. For example: if we look at the given samples in the question, `(1, 1)` for filter lies on `(1, 1)` of input and `(0, 0)` on `(0, 0)`. However, for input `(1, 0)`, the filter goes out of the input so, the input indices corresponding to `(0, 0)` would be `(-1, 0)`. Since this is out of the input, instead of explicitly multiplying by 0, we just ignore them using the above if condition. So, the performance should be better than actually iteratively adding zeros.

---
**NOTE**

This implementation will have the same effects as actually adding the 0s for padding. Just that they are not explicitly added. 
---

3. For normalisation: Again, focus was on efficiency, so the sum of the kernel is passed instead of a normalised filter and then the calculated value is normalised instead of filter. 

4. For part 2, evert channel is indeed calculated differently. This is done as follows. Every index in the 1d array represents an `(i,j,k)` in the 3d where `k` takes values of `(0,1,2)` depending on the channel. So calculating the channels separately is just finding the correspoding values that need to fit into the filter. This is done by a bit of index manipulation shown in the `sharpen` function.


# Recorded Times

## Part 1

```
base) ujain@gpu-comp-102:~/comp3231/assign_1$ ./part1
Reading image_grey.txt... 
5046272 entries imported
Reading image_grey.txt... 
5046272 entries imported

CPU Implementation
Elapsed time: 379.442ms
Display result: 33
22
22
22
22
22
22
23
19
20
.
.
.
115
122
133
124
119
115
116
104
108
160
Saving data to grey_result_CPU.txt... 
5046272 entries saved

GPU Implementation
Kernel Elapsed time: 1.24048ms
Elapsed time: 249.446ms
Display result: 33
22
22
22
22
22
22
23
19
20
.
.
.
115
122
133
124
119
115
116
104
108
160
Saving data to grey_result_GPU.txt... 
5046272 entries saved
```

## Part 2

```
(base) ujain@gpu-comp-102:~/comp3231/assign_1$ ./part2
Reading image_color.txt... 
15552000 entries imported
Reading image_color.txt... 
15552000 entries imported

CPU Implementation
Elapsed time: 3915.17ms
Display result: (original -> result)
4 -> 1
11 -> 4
19 -> 7
4 -> 1
11 -> 6
19 -> 10
3 -> 2
12 -> 7
19 -> 11
3 -> 1
.
.
.
80 -> 49
7 -> 2
66 -> 39
80 -> 48
7 -> 1
66 -> 34
82 -> 43
7 -> 1
66 -> 25
82 -> 32
Saving data to color_result_CPU.txt... 
5184000 entries saved

GPU Implementation
Kernel Elapsed time: 20.9808ms
Elapsed time: 362.404ms
Display result: (original -> result)
4 -> 1
11 -> 4
19 -> 7
4 -> 1
11 -> 6
19 -> 10
3 -> 2
12 -> 7
19 -> 11
3 -> 1
.
.
.
80 -> 49
7 -> 2
66 -> 39
80 -> 48
7 -> 1
66 -> 34
82 -> 43
7 -> 1
66 -> 25
82 -> 32
Saving data to color_result_GPU.txt... 
5184000 entries saved
```
