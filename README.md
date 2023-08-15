### Summary

最终成绩 237 ms

转为trt fp16 ~600ms

大量torch蚊子腿优化 ~470ms

减少step ~290ms

groupnorm plugin with fp16，shared memory，internel batch=2 ~237ms

### 未完成项目

nsys的使用

cudagraph

multi stream execution

### 尝试过失败项目

PTQ，QAT

flash attention plugin

merge two loops in one trt engine

