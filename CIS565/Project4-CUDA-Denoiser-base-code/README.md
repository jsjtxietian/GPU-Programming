CUDA Denoiser For CUDA Path Tracer
==================================

完成了基础部分：
1. add UI controls to your project - we've done this for you in this base code, but see Base Code Tour
2》 implement G-Buffers for normals and positions and visualize them to confirm (see Base Code Tour)
3. implement the A-trous kernel and its iterations without weighting and compare with a a blur applied from, say, GIMP or Photoshop
4. use the G-Buffers to preserve perceived edges（论文看不明白，借鉴了Linda的代码）
