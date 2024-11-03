#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration);
void showGBuffer(uchar4 *pbo, const int mod = 0); // 0 => pos, 1 => normal
void showImage(uchar4 *pbo, int iter);
