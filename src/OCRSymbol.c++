/*
 *  This file is part of vobsub2srt
 *
 *  Copyright (C) 2026 Bastiaan Stougie <wififreedom2026@protonmail.com>
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "OCRSymbol.h++"

#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "generic_exception.h++"

OCRSymbol::OCRSymbol(
    const char* const utf8_symbol)
    : priv_utf8_symbol(),
      itd_bbox_vec(),
      priv_italic_confidence(DEFAULT_CONFIDENCE) {

  if (utf8_symbol == NULL) {
    std::cerr << "NULL symbol" << std::endl;
    throw std::invalid_argument("OCRSymbol(): NULL symbol");
  }

  priv_utf8_symbol = std::string(utf8_symbol);
}

OCRSymbol::OCRSymbol(
    const std::string& utf8_symbol)
    : priv_utf8_symbol(utf8_symbol),
      itd_bbox_vec(),
      priv_italic_confidence(DEFAULT_CONFIDENCE) {
}

bool
OCRSymbol::is_one_of(
    const char* const set) const {
  // TODO if necessary make compatible with UTF-8
  return (std::strchr(set, priv_utf8_symbol[0]) != NULL);
}

void
OCRSymbol::build_stats(
    const cv::Mat& img,
    TextStats& stats) const {
  if (itd_bbox_vec.size() == 1) {
    const cv::Rect& bbox = itd_bbox_vec[0];
    stats.symbol_add(
	priv_utf8_symbol,
	bbox,
	bbox_top_row_left_pixel_pos(img, bbox) -
	bbox_bottom_row_left_pixel_pos(img, bbox),
	bbox_top_row_right_pixel_pos(img, bbox) -
	bbox_bottom_row_right_pixel_pos(img, bbox));
  }
}

void
OCRSymbol::bbox_assign(
    const cv::Rect& bbox,
    bool test_only) {
  if (itd_bbox_vec.size()) {
    throw generic_exception("OCRSymbol::bbox_assign: bbox already assigned");
  }
  if (test_only) {
    return;
  }
  itd_bbox_vec.emplace_back(bbox);
}

void
OCRSymbol::bboxes_assign(
    const std::vector<cv::Rect>& bboxes,
    bool test_only) {
  if (itd_bbox_vec.size()) {
    throw generic_exception("OCRSymbol::bboxes_assign: bbox already assigned");
  }
  if (test_only) {
    return;
  }
  itd_bbox_vec = bboxes;
}

void
OCRSymbol::bboxes_remove() {
  itd_bbox_vec.clear();
}

void
OCRSymbol::assign_confidence(
    const cv::Mat& img,
    const TextStats& stats) {

  // punctuation
  if (priv_utf8_symbol[0] == '-') {
    // occurs as a word at the begin of a line
    priv_italic_confidence = -1;
    return;
  }
  if (strchr(".,'\"", priv_utf8_symbol[0])) {
    // ignore, inconclusive results
    return;
  }

  if (itd_bbox_vec.size() == 1) {
    cv::Rect& bbox = itd_bbox_vec[0];

    // TODO see if we can base the constants -1.8 on something (bbox width?)

    // Symbols mis-identified by OCR should not be assigned a confidence value.
    // Use symbol stats to try te detect mis-identified symbols.
    const SymbolStat* stat = stats.symbol_stat(priv_utf8_symbol);
    if (!stat) {
      // No info, do not assign confidence value.
      return;
    }

    const std::optional<float> avg_width_opt = stat->width_avg();
    if (!avg_width_opt.has_value()) {
      // No info, do not assign confidence value.
      return;
    }
    if (bbox.width < (avg_width_opt.value() - 1.8)) {
      // OCR possibly misidentified the symbol.
      return;
    }

    const std::optional<int> min_height_opt = stat->height_min();
    if (!min_height_opt.has_value()) {
      // No info, do not assign confidence value.
      return;
    }
    if (bbox.height < (min_height_opt.value())) {
      // OCR possibly misidentified the symbol.
      return;
    }

    const std::optional<float> avg_l_opt = stat->l_avg();
    if (!avg_l_opt.has_value()) {
      // No info, do not assign confidence value.
      return;
    }
    const int top_l = bbox_top_row_left_pixel_pos(img, bbox);
    const int bottom_l = bbox_bottom_row_left_pixel_pos(img, bbox);
    if (top_l == -1 || bottom_l == -1) {
      // Bad bbox, possibly a split combined one.
      return;
    }
    const float val_l = (top_l - bottom_l) - avg_l_opt.value();

    const std::optional<float> avg_r_opt = stat->r_avg();
    if (!avg_r_opt.has_value()) {
      // No info, do not assign confidence value.
      return;
    }
    const int top_r = bbox_top_row_right_pixel_pos(img, bbox);
    const int bottom_r = bbox_bottom_row_right_pixel_pos(img, bbox);
    if (top_r == -1 || bottom_r == -1) {
      // Bad bbox, possibly a split combined one.
      return;
    }
    const float val_r = (top_r - bottom_r) - avg_r_opt.value();

    if (val_l > 0.05 /* error margin */ || val_r > 0.05 /* error margin */) {
      priv_italic_confidence = (1 + std::max(val_l, val_r)) * bbox.height;
    }
    else {
      priv_italic_confidence = -bbox.height;
    }
  }
  else if (itd_bbox_vec.size() == 0) {
    priv_italic_confidence = 11;
  }
  else if (itd_bbox_vec.size() > 1) {
    priv_italic_confidence = 5;
  }
}

void
OCRSymbol::dump(
    std::ostream& os) const {
  os << "      utf8_symbol: " << priv_utf8_symbol;
  for (std::size_t i = 1; const auto& bbox : itd_bbox_vec) {
    os << ", bbox " << i << ": (" <<
      bbox.x << "," <<
      bbox.y << ")-(" <<
      (bbox.x + bbox.width) << "," <<
      (bbox.y + bbox.height) << ")";
    i++;
  }
  os << ", ic: " << priv_italic_confidence <<
    std::endl;
}

void
OCRSymbol::itd_bboxes_draw(
  const cv::Mat& img,
  const cv::Rect& line_bbox,
  unsigned char grayscale_color) const {

  for (const auto& bbox : itd_bbox_vec) {
    cv::rectangle(
	img,
	cv::Rect(
	  bbox.x - line_bbox.x,
	  bbox.y - line_bbox.y,
	  bbox.width,
	  bbox.height),
	cv::Scalar(grayscale_color),
	1);
  }
}

