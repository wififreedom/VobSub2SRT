/*
 *  This file is part of vobsub2srt.
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

#include "OCRSubtitle.h++"
#include "Replacements.h++"

//#include <opencv2/core.hpp>
#include <tesseract/baseapi.h>

#ifndef OCRSUBTITLES_HXX
#define OCRSUBTITLES_HXX

class OCRSubtitles {

private:
  class SubtitleInfo {

  public:
    SubtitleInfo(
	const std::size_t subtitle_number,
	const uint32_t start_pts,
	const uint32_t end_pts,
	const unsigned char* const sp_image,
	const unsigned sp_width,
	const unsigned sp_height,
	const unsigned sp_stride);

    std::size_t subtitle_number;
    uint32_t start_pts;
    uint32_t end_pts;

    // original image after conversion to white text on black background for
    // ocr
    cv::Mat ocr_img;
    std::vector<cv::Rect> ocr_line_bbox_vec;

    // original image after conversion to white text on black background for
    // italic detection
    cv::Mat itd_img;
    std::vector<cv::Rect> itd_line_bbox_vec;
  };

public:
  OCRSubtitles(
      const std::string& subname);

  bool
  append(
      const std::size_t subtitle_number,
      const uint32_t start_pts,
      const uint32_t end_pts,
      const unsigned char* const sp_image,
      const unsigned sp_width,
      const unsigned sp_height,
      const unsigned sp_stride);

  void
  do_ocr(
      tesseract::TessBaseAPI& tess_base_api,
      const std::size_t ocr_batch_size,
      const bool show);

  void
  correct_ocr(
      const Replacements& replacements);

  void
  detect_italic();

  void
  write_srt(
      std::ostream& os,
      const int base_duration,
      const int chars_per_sec);

  void
  bboxes_assign(
      TextStats * const stats);

  void
  build_stats(
      TextStats& stats) const;

  void
  bboxes_remove();

private:
  void
  do_ocr(
      tesseract::TessBaseAPI& tess_base_api,
      const std::size_t batch_begin_i,
      const std::size_t batch_end_i);

  cv::Mat
  batch_prepare(
      const std::size_t batch_begin_i,
      const std::size_t batch_end_i,
      std::vector<std::vector<cv::Rect>>& ocr_line_bbox_vec_vec,
      cv::Mat& itd_batch_img,
      std::vector<std::vector<cv::Rect>>& itd_line_bbox_vec_vec) const;

  void
  batch_ocr(
      tesseract::TessBaseAPI& tess_base_api,
      const cv::Mat& batch_ocr_img,
      const std::vector<std::vector<cv::Rect>>& ocr_line_bbox_vec_vec,
      const std::size_t batch_begin_subtitle_i,
      const std::size_t batch_begin_line_i,
      const cv::Mat& itd_batch_img,
      const std::vector<std::vector<cv::Rect>>& itd_line_bbox_vec_vec,
      std::vector<std::vector<OCRLine>>& subtitle_line_vec) const;

  void
  batch_ocr(
      tesseract::TessBaseAPI& tess_base_api,
      const std::size_t batch_begin_subtitle_i,
      const std::size_t batch_end_subtitle_i);

  cv::Mat
  recover_prepare(
      const std::size_t subtitle_i,
      const std::size_t line_i,
      const std::size_t fill_subtitle_i,
      const std::size_t fill_line_i,
      std::vector<std::vector<cv::Rect>>& ocr_line_bbox_vec_vec,
      cv::Mat& itd_batch_img,
      std::vector<std::vector<cv::Rect>>& itd_line_bbox_vec_vec);

  OCRLine
  recover_line(
      tesseract::TessBaseAPI& tess_base_api,
      const std::size_t subtitle_i,
      const std::size_t line_i,
      const std::size_t fill_subtitle_i,
      const std::size_t fill_line_i);

  void
  recover_subtitle(
      tesseract::TessBaseAPI& tess_base_api,
      const std::size_t subtitle_i,
      const bool show);

  void
  recover(
      tesseract::TessBaseAPI& tess_base_api,
      const bool show);

  std::ostream&
  write(
      std::ostream& os) const;

  void
  read(
      std::istream& is);

private:
  std::string subname;

  std::vector<SubtitleInfo> subtitle_info_vec;

  std::vector<OCRSubtitle> subtitle_vec;
};

#endif // OCRSUBTITLES_HXX

