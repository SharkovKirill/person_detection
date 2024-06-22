#!/bin/bash

pushd subsampled_size_0.15_Or_AtLeastOne

mv VisDrone/VisDrone2019-DET-val VisDrone/valid
mv VisDrone/VisDrone2019-DET-train VisDrone/train
mv VisDrone/VisDrone2019-DET-test-dev VisDrone/test

popd