#include "cocoeval/cocoeval.h"
#include "box_iou_rotated/box_iou_rotated.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("COCOevalAccumulate", &COCOeval::Accumulate, "COCOeval::Accumulate");
    m.def(
        "COCOevalEvaluateImages",
        &COCOeval::EvaluateImages,
        "COCOeval::EvaluateImages");
    pybind11::class_<COCOeval::InstanceAnnotation>(m, "InstanceAnnotation")
        .def(pybind11::init<uint64_t, double, double, bool, bool>());
    pybind11::class_<COCOeval::ImageEvaluation>(m, "ImageEvaluation")
        .def(pybind11::init<>());
}

TORCH_LIBRARY(yolox, m) {
  m.def("box_iou_rotated", &yolox::box_iou_rotated);
}
