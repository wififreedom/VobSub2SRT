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

#include "OCRSubtitles.h++"

#include "generic_exception.h++"
#include "debug.h++"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <format>

static uchar
img_mean_nonzero(
    const cv::Mat& img) {
  uint64_t total = 0;
  uint64_t count = 0;
  for (int row = 0; row < img.rows; row++) {
    for (int col = 0; col < img.cols; col++) {
      const uchar value = img.at<uchar>(row, col);
      if (value) {
	total += value;
	count++;
      }
    }
  }
  if (!count) {
    return 0;
  }
  return (uchar) (total / count);
}

static void
img_histogram(
    const cv::Mat& img,
    std::vector<int>& color_histogram) {
  color_histogram.clear();
  color_histogram.resize(256, 0);

  for (int row = 0; row < img.rows; row++) {
    for (int col = 0; col < img.cols; col++) {
      const uchar value = img.at<uchar>(row, col);
      color_histogram[static_cast<std::size_t>(value)]++;
    }
  }
}

std::ostream&
dump_histogram(
    std::ostream& os,
    std::vector<int>& color_histogram) {
  for (std::size_t i = 0; i < color_histogram.size(); i++) {
    if (color_histogram[i]) {
      os << "  " << std::format("{:4}", i) << ": " <<
	std::format("{:5}", color_histogram[i]) << std::endl;
    }
  }
  return os;
}

OCRSubtitles::SubtitleInfo::SubtitleInfo(
    const std::size_t subtitle_number,
    const uint32_t start_pts,
    const uint32_t end_pts,
    const unsigned char* const sp_image,
    const unsigned sp_width,
    const unsigned sp_height,
    const unsigned sp_stride)
    : subtitle_number(subtitle_number),
      start_pts(start_pts),
      end_pts(end_pts),
      bw_img(std::move(cv::Mat(
	      static_cast<int>(sp_height),
	      static_cast<int>(sp_width),
	      CV_8UC1,
	      (void*) sp_image,
	      static_cast<std::size_t>(sp_stride)).clone())),
      bw_line_bbox_vec() {

  // Convert grayscale to white text on black background.
  // NOTA BENE: if text has grayscale color value < threshold, all text
  // will disappear.

  std::vector<int> histogram;
  img_histogram(bw_img, histogram);
  std::size_t min = INT_MAX;
  std::size_t count = 0;
  // skip 0 = transparent
  for (std::size_t i = 1; i < histogram.size(); i++) {
    if (histogram[i]) {
      min = std::min(min, i);
      count++;
    }
  }
  int threshold = 128;
  if (count == 1) {
    threshold = 255;
  }
  else if (count == 2 || count == 3) {
    // 2 or 3 colors, remove only darkest by thresholding
    // in order to not lose too much detail
    threshold = static_cast<int>(min);
  }

  cv::threshold(bw_img, bw_img, threshold, 255, cv::THRESH_BINARY);

  get_text_line_bboxes(bw_img, bw_line_bbox_vec);
}

OCRSubtitles::OCRSubtitles(
    const std::string& subname)
    : subname(subname),
      subtitle_info_vec(),
      subtitle_vec() {
}

bool
OCRSubtitles::append(
    const std::size_t subtitle_number,
    const uint32_t start_pts,
    const uint32_t end_pts,
    const unsigned char* const sp_image,
    const unsigned sp_width,
    const unsigned sp_height,
    const unsigned sp_stride) {

  // Tesseract performs slightly better with larger amounts of text,
  // so store subtitle information, then process in batches in
  // combined images.

  SubtitleInfo si(
      subtitle_number,
      start_pts,
      end_pts,
      sp_image,
      sp_width,
      sp_height,
      sp_stride);

  const bool has_lines = !si.bw_line_bbox_vec.empty();

  if (has_lines) {
    subtitle_info_vec.emplace_back(std::move(si));
  }

  if (debug || subtitle_number == debug_subtitle_number) {
    const cv::Mat sp_img(
	  static_cast<int>(sp_height),
	  static_cast<int>(sp_width),
	  CV_8UC1,
	  (void*) sp_image,
	  static_cast<std::size_t>(sp_stride));
    if (debug_ext.size()) {
      std::stringstream ss;
      ss << subname << "-" << subtitle_number << "-spudec." << debug_ext;
      cv::imwrite(ss.str(), sp_img);
      std::cerr << "subtitle " << subtitle_number << ": image saved as " << ss.str() << std::endl;
    }
    std::vector<int> sp_img_histogram;
    img_histogram(sp_img, sp_img_histogram);
    std::cerr << "subtitle " << subtitle_number << ", spudec image histogram (color value: count)" << std::endl;
    dump_histogram(std::cerr, sp_img_histogram);
    const int mean = img_mean_nonzero(sp_img);
    std::cerr << "subtitle " << subtitle_number << ", mean_nonzero " << mean << std::endl;
  }

  return has_lines;
}

void
OCRSubtitles::do_ocr(
    tesseract::TessBaseAPI& tess_base_api,
    const std::size_t ocr_batch_size,
    const bool show) {

  // TODO: more consistent number of lines per image, not number of subtitles
  const std::size_t batch_size = ocr_batch_size > 0 ? ocr_batch_size : 1;

  if (debug) {
    std::cerr << "batch OCR: batch size " << batch_size << " subtitles." << std::endl;
  }

  for (std::size_t batch_i = 0; batch_i < subtitle_info_vec.size(); batch_i += batch_size) {
    const std::size_t batch_end_i =
      std::min(batch_i + batch_size, subtitle_info_vec.size());

    do_ocr(
	tess_base_api,
	batch_i,
	batch_end_i);

    if (show) {
      for (std::size_t subtitle_i = batch_i; subtitle_i < batch_end_i; subtitle_i++) {
	for (std::size_t line_i = 0; line_i < subtitle_vec[subtitle_i].num_lines(); line_i++) { 
	  std::cout << "Subtitle " << (subtitle_i + 1) << ": ";
	  subtitle_vec[subtitle_i].write_line(std::cout, line_i);
	}
      }
    }
  }

  // If there were any problems performing OCR on some subtitle lines,
  // attempt to recover.
  recover(tess_base_api, show);
}

void
OCRSubtitles::correct_ocr(
    const Replacements& replacements) {
  std::stringstream ss;
  {
    // reserve space
    std::string tmp;
    // string remains empty, but is pre-allocated
    tmp.reserve(64 * 1024);
    ss.str(std::move(tmp));
  }
  write(ss);

  std::istringstream iss;
  iss.str(replacements.replace(ss.str()));
  read(iss);
}

void
OCRSubtitles::detect_italic() {
  // Strategy:
  // - Perform OCR on all subtitles (only once). This has already been
  //   done.
  // - Perform italic detection in multiple phases:
  //   - Phase 1: Assign high-confidence bboxes to symbols.
  //     - Iterate over all subtitles: assign only bboxes to symbols if highly
  //       confident that they are correct:
  //       - correct number of word bboxes for the line
  //       - no invalid word bbox
  //       - no overlapping word bbox,
  //       - no invalid symbol bbox,
  //       - no overlapping symbol bbox,
  //       - all symbol bboxes inside the word bbox,
  //       - correct number of symbol bboxes for the number of symbols in 
  //         the word
  //   - Phase 2: build statistics.
  //     - Gather min, average and max width and height for each symbol
  //       that has a bbox assigned to it. Gather min, average, and max
  //       values for inter-word spacing and inter-symbol spacing within
  //       words.
  //     - Despite the effort in phase one, incorrect data may be used
  //       for the statistics, if:
  //       - OCR incorrectly identified a character
  //       - multiple issues with bboxes occur in the same word, for example,
  //         both an additional bbox (overlapping bboxes), and a single bbox
  //         for two symbols. This leads to a correct number of bboxes,
  //         which is accepted without further checks, because in this
  //         phase there are no statistics to base any checks on.
  //       For that reason, check for and exclude anything suspicious that
  //       could result in bad statistics, such as bounding boxes that can't
  //       be the correct size for a symbol, word spacing that can't be
  //       correct, etc.
  //   - Phase 3: remove all assigned bboxes from symbols.
  //     - With the statistics, it may be posible to arrive at a better result
  //       for bbox assignment. For example, based on the statistics about
  //       inter-word spacing, we can possibly better correct detected
  //       inaccurate word bboxes. This affects which bboxes are assigned to
  //       symbols. So wipe the previous result and start over.
  //   - Phase 4: use statistics to assign bboxes to symbols.
  //     - In this phase, we try to assign a bbox to as many symbols as
  //       possible, not just if there is high confidence.
  //   - Phase 5: detect italic
  //     - Based on the assigned bboxes, detect italic 

  TextStats text_stats;

  // Phase 1
  bboxes_assign(NULL);

  // Phase 2
  build_stats(text_stats);

  if (debug || debug_subtitle_number > 0) {
    text_stats.dump(std::cerr);
  }
  
  // Phase 3
  bboxes_remove();

  // Phase 4
  bboxes_assign(&text_stats);

  // Phase 5
  for (auto& it : subtitle_vec) {
    it.detect_italic(text_stats);
  }
}

void
OCRSubtitles::write_srt(
    std::ostream& os,
    const int base_duration,
    const int chars_per_sec) {
  for (std::size_t i = 0; i < subtitle_vec.size(); i++) {
    subtitle_vec[i].write_srt(
	os,
	base_duration,
	chars_per_sec,
	((i + 1) < subtitle_vec.size()) ? 
	  std::optional(subtitle_vec[i + 1].start_pts_get()) : std::nullopt);
  }
}

void
OCRSubtitles::bboxes_assign(
    TextStats* const stats) {
  for (auto& it : subtitle_vec) {
    it.bboxes_assign(subname, stats);
  }
}

void
OCRSubtitles::build_stats(
    TextStats& stats) const {
  for (const auto& it : subtitle_vec) {
    it.build_stats(stats);
  }
}

void
OCRSubtitles::bboxes_remove() {
  for (auto& it : subtitle_vec) {
    it.bboxes_remove();
  }
}

void
OCRSubtitles::do_ocr(
    tesseract::TessBaseAPI& tess_base_api,
    const std::size_t batch_begin_subtitle_i,
    const std::size_t batch_end_subtitle_i) {

  std::string msg;

  try {
    batch_ocr(
	tess_base_api,
	batch_begin_subtitle_i,
	batch_end_subtitle_i);

    return;
  }
  catch (const generic_exception& e) {
    msg = e.what();
  }

  // Could be out of sync due to OCR not recognizing any text in an entire line.
  // Try to repair by splitting the batch, if necessary repeatedly.
  const std::size_t batch_size = batch_end_subtitle_i - batch_begin_subtitle_i;
  if (batch_size > 1) {
    const std::size_t batch_split_subtitle_i =
      batch_begin_subtitle_i + (batch_size / 2);

    do_ocr(tess_base_api, batch_begin_subtitle_i, batch_split_subtitle_i);

    do_ocr(tess_base_api, batch_split_subtitle_i, batch_end_subtitle_i);
  }
  else {
    // The problem is in (one of) the lines of this one subtitle.
    // Insert an empty subtitle for now, and try to recover later, when there are
    // more successful OCR results to work with for recovery.

    const SubtitleInfo& si = subtitle_info_vec[batch_begin_subtitle_i];

    std::vector<OCRLine> line_vec;
    subtitle_vec.emplace_back(
	si.subtitle_number,
	si.start_pts,
	si.end_pts,
	line_vec);

    std::cerr << msg << ", will attempt to recover" << std::endl;
  }
}

cv::Mat
OCRSubtitles::batch_prepare(
    const std::size_t batch_begin_i,
    const std::size_t batch_end_i,
    std::vector<std::vector<cv::Rect>>& combined_line_bbox_vec_vec) const {

  // Gather information required to calculate combined image dimensions:
  // - total number of lines
  // - maximum line bbox width
  // - maximum line bbox height
  int line_count = 0;
  int line_max_width = 0;
  int line_max_height = 0;
  for (std::size_t si_i = batch_begin_i; si_i < batch_end_i; si_i++) {
    if (debug || ((si_i + 1) == debug_subtitle_number)) {
      std::cerr << "Batch OCR: subtitle " << (si_i + 1) << ", # lines: " <<
	subtitle_info_vec[si_i].bw_line_bbox_vec.size() << std::endl;
    }
    for (const auto& lb_it : subtitle_info_vec[si_i].bw_line_bbox_vec) {
      line_count++;
      line_max_width = std::max(line_max_width, lb_it.width);
      line_max_height = std::max(line_max_height, lb_it.height);
    }
  }

  if (debug) {
    std::cerr << "batch OCR: batch has " << line_count << " lines." << std::endl;
  }

  // NUM_BORDER_PIXELS is used for the vertical distance between lines as well.
  const int combined_img_width =
    NUM_BORDER_PIXELS /* left border */
    + line_max_width
    + NUM_BORDER_PIXELS /* right border */;
  const int combined_img_height =
    NUM_BORDER_PIXELS /* top border */
    + (line_count * line_max_height) /* lines */
    + ((line_count - 1) * NUM_BORDER_PIXELS) /* spacing between lines */
    + NUM_BORDER_PIXELS /* bottom border */;

  // New image with all white pixels
  cv::Mat combined_img(
      combined_img_height,
      combined_img_width,
      CV_8UC1,
      cv::Scalar(255));

  // Copy text lines into image, left-aligned
  {
    std::size_t line_i = 0;
    for (std::size_t si_i = batch_begin_i; si_i < batch_end_i; si_i++) {
      combined_line_bbox_vec_vec.emplace_back();
      for (const auto& lb_it : subtitle_info_vec[si_i].bw_line_bbox_vec) {
	const cv::Rect target_bbox(
	    NUM_BORDER_PIXELS,
	    NUM_BORDER_PIXELS + line_i * (line_max_height + NUM_BORDER_PIXELS),
	    lb_it.width,
	    lb_it.height);
	combined_line_bbox_vec_vec[si_i - batch_begin_i].emplace_back(target_bbox);

	// Copy line from bw_img into combined image, inverted to have black
	// text on white background.
	cv::Mat src(subtitle_info_vec[si_i].bw_img, lb_it);
	cv::Mat dst(combined_img, target_bbox);
	cv::bitwise_not(src, dst);

	line_i++;
      }
    }
  }

  if (debug ||
      (((batch_begin_i + 1) <= debug_subtitle_number) &&
       ((batch_end_i + 1) > debug_subtitle_number))) {
    if (debug_ext.size()) {
      std::stringstream ss;
      ss << subname << "-" << "combined-" << (batch_begin_i + 1) << "-" << (batch_end_i - batch_begin_i) << "." << debug_ext;
      cv::imwrite(ss.str(), combined_img);
      std::cerr << "batch OCR: batch image saved as " << ss.str() << std::endl;
    }
  }

  return combined_img;
}

void
OCRSubtitles::batch_ocr(
    tesseract::TessBaseAPI& tess_base_api,
    const cv::Mat& combined_img,
    const std::vector<std::vector<cv::Rect>>& combined_line_bbox_vec_vec,
    const std::size_t batch_begin_subtitle_i,
    const std::size_t batch_begin_line_i,
    std::vector<std::vector<OCRLine>>& subtitle_line_vec) const {

  if (combined_line_bbox_vec_vec.size() == 0) {
    // nothing to do
    return;
  }

  if (batch_begin_line_i >= combined_line_bbox_vec_vec[0].size()) {
    throw generic_exception("OCRSubtitles::batch_ocr: batch begin line index out of bounds");
  }

  tess_base_api.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
  tess_base_api.SetImage(
      combined_img.data,
      combined_img.cols,
      combined_img.rows,
      1 /* bytes per pixel */,
      static_cast<int>(combined_img.step));
  tess_base_api.Recognize(0);

  std::unique_ptr<tesseract::ResultIterator> ri(tess_base_api.GetIterator());
  if (!ri) {
    throw generic_exception("ocr: could not get tesseract ResultIterator");
  }

  std::size_t rel_si_i = 0;
  std::size_t rel_line_i = 0; /* combined_line_bbox_vec_vec[rel_si_i] index */
  std::vector<OCRLine> line_vec;
  std::vector<OCRWord> word_vec;
  std::vector<cv::Rect> word_bbox_vec;
  std::vector<cv::Rect> symbol_bbox_vec;
  std::size_t subtitle_number = batch_begin_subtitle_i + 1;
  std::size_t line_number = batch_begin_line_i + 1;

  do {
    bool first = true;

    // skip subtitles with blank images
    while (rel_si_i < combined_line_bbox_vec_vec.size() &&
	combined_line_bbox_vec_vec[rel_si_i].size() == 0) {
      rel_si_i++;
    }

    if (rel_si_i == combined_line_bbox_vec_vec.size()) {
      std::stringstream ss;
      ss << "subtitle " << subtitle_number <<
	", line " << line_number <<
	": ocr: more lines than expected for batch";
      throw generic_exception(ss.str());
    }

    do {
      std::vector<OCRSymbol> symbol_vec;

      do {
	if (!first) {
	  ri->Next(tesseract::RIL_SYMBOL);
	}
	else {
	  first = false;
	}

	// If there is no more text in the image, GetUTF8Text returns NULL.
	// This can only happen if Tesseract has not recognized any text in the
	// image. In all other cases ri->Next returns false and the loop exits.
	std::unique_ptr<char[]> symbol(ri->GetUTF8Text(tesseract::RIL_SYMBOL));
	if (!symbol) {
	  std::stringstream ss;
	  ss << "subtitle " << subtitle_number <<
	    ", line " << line_number <<
	    ": ocr: no text recognized in the image";
	  throw generic_exception(ss.str());
	}

	// work around bug in tesseract, which recognizes 'I' as '|'
	// '|' is very unlikely to occur in subtitles.
	if (symbol[0] == '|')
	  symbol[0] = 'I';

	// Tesseract's observed behaviour for the subtitles is equivalent
	// to:
	// - it calculates bounding boxes in some way, most perfectly
	//   accurate, but some wildly inaccurate.
	// - it then sorts the bounding boxes by something like the middle
	//   of the bbox (sometimes they are not ordered by x coordinate),
	//   inserting inaccurate bboxes (if any) in the wrong place.
	// - it then returns the bounding box at index i for the symbol at
	//   index i.
	// - the returned bounding box may therefore be of a different
	//   symbol.
	// For this reason, we do not immediately assign bounding boxes to
	// words and symbols, but make a list first, then check and improve
	// the bounding boxes by using a list of bounding boxes determined
	// with opencv contour detection, then assign the bounding boxes to
	// words and symbols.
	//
	// The returned bottom and right of the bounding box are exclusive, just
	// like for the contour bboxes we determine later.
	//
	// See tesseract/pageiterator.h, comment about "Coordinate system".
	
	symbol_vec.emplace_back(symbol.get());

	int left = 0, top = 0, right = 0, bottom = 0;
	if (!ri->BoundingBox(tesseract::RIL_SYMBOL,
	      &left, &top, &right, &bottom)) {
	  std::stringstream ss;
	  ss << "subtitle " << subtitle_number <<
	    ", line " << line_number <<
	    ": ocr: could not get symbol bounding box";
	  throw generic_exception(ss.str());
	}
	symbol_bbox_vec.emplace_back(
	  cv::Rect(left, top, right - left, bottom - top));

      } while (!ri->IsAtFinalElement(tesseract::RIL_WORD, tesseract::RIL_SYMBOL));

      word_vec.emplace_back(
	  subtitle_number,
	  line_number,
	  word_vec.size() + 1,
	  std::move(symbol_vec));

      int left = 0, top = 0, right = 0, bottom = 0;
      if (!ri->BoundingBox(tesseract::RIL_WORD,
	    &left, &top, &right, &bottom)) {
	std::stringstream ss;
	ss << "subtitle " << subtitle_number <<
	  ", line " << line_number <<
	  ": ocr: could not get word bounding box";
	throw generic_exception(ss.str());
      }
      word_bbox_vec.emplace_back(
	cv::Rect(left, top, right - left, bottom - top));
    } while (!ri->IsAtFinalElement(tesseract::RIL_TEXTLINE, tesseract::RIL_SYMBOL));

    // To avoid making a copy, the OCRLine constructor swaps word_vec,
    // word_bbox_vec, and symbol_bbox_vec with empty vectors, so they are
    // empty after this.
    line_vec.emplace_back(
	subtitle_number,
	line_number,
	combined_img,
	word_vec,
	combined_line_bbox_vec_vec[rel_si_i][rel_line_i],
	word_bbox_vec,
	symbol_bbox_vec);

    if ((rel_line_i + 1) == combined_line_bbox_vec_vec[rel_si_i].size()) {
      std::vector<OCRLine> tmp;
      tmp.swap(line_vec);
      // line_vec is now empty, move tmp into subtitle_line_vec
      subtitle_line_vec.emplace_back(
	  std::move(tmp));

      rel_si_i++;
      subtitle_number++;
      rel_line_i = 0;
      line_number = 1;
    }
    else {
      rel_line_i++;
      line_number++;
    }

  } while (ri->Next(tesseract::RIL_SYMBOL));

  if (rel_si_i != combined_line_bbox_vec_vec.size()) {
    std::stringstream ss;
    ss << "subtitle " << (batch_begin_subtitle_i + rel_si_i + 1) <<
      ", line " << (rel_line_i + 1) <<
      ": ocr: fewer lines than expected for batch";
    throw generic_exception(ss.str());
  }
}

void
OCRSubtitles::batch_ocr(
    tesseract::TessBaseAPI& tess_base_api,
    const std::size_t batch_begin_subtitle_i,
    const std::size_t batch_end_subtitle_i) {

  std::vector<std::vector<cv::Rect>> combined_line_bbox_vec_vec;
  std::vector<std::vector<OCRLine>> subtitle_line_vec;

  const cv::Mat combined_img(batch_prepare(
      batch_begin_subtitle_i,
      batch_end_subtitle_i,
      combined_line_bbox_vec_vec));

  batch_ocr(
      tess_base_api,
      combined_img,
      combined_line_bbox_vec_vec,
      batch_begin_subtitle_i,
      0,
      subtitle_line_vec);

  for (std::size_t i = 0; i < subtitle_line_vec.size(); i++) {
    const SubtitleInfo& si = subtitle_info_vec[batch_begin_subtitle_i + i];
    subtitle_vec.emplace_back(
	si.subtitle_number,
	si.start_pts,
	si.end_pts,
	subtitle_line_vec[i]);
  }
}

cv::Mat
OCRSubtitles::recover_prepare(
    const std::size_t subtitle_i,
    const std::size_t line_i,
    const std::size_t filler_subtitle_i,
    const std::size_t filler_line_i,
    std::vector<std::vector<cv::Rect>>& combined_line_bbox_vec_vec) {

  const SubtitleInfo& si = subtitle_info_vec[subtitle_i];
  const SubtitleInfo& filler_si = subtitle_info_vec[filler_subtitle_i];

  const cv::Rect& bbox = si.bw_line_bbox_vec[line_i];
  const cv::Rect& filler_bbox = filler_si.bw_line_bbox_vec[filler_line_i];
  const int max_height = std::max(bbox.height, filler_bbox.height);

  // TODO if possible, align the text of filler line and line vertically

  const int combined_img_width =
    NUM_BORDER_PIXELS /* left border */
    + filler_bbox.width
    + NUM_BORDER_PIXELS /* horizontal spacing between lines of text */
    + bbox.width
    + NUM_BORDER_PIXELS /* horizontal spacing between lines of text */
    + filler_bbox.width
    + NUM_BORDER_PIXELS /* right border */;
  const int combined_img_height =
    NUM_BORDER_PIXELS /* top border */
    + max_height
    + NUM_BORDER_PIXELS /* bottom border */;

  // New image with all white pixels
  cv::Mat combined_img(
      combined_img_height,
      combined_img_width,
      CV_8UC1,
      cv::Scalar(255));

  // Copy filler line in front of and behind the line to recoveer.
  for (std::size_t i = 0; i < 2; i++) {
    int x_pos = NUM_BORDER_PIXELS + i * (
	filler_bbox.width
	+ NUM_BORDER_PIXELS
	+ bbox.width
	+ NUM_BORDER_PIXELS);

    const cv::Rect target_bbox(
	x_pos,
	NUM_BORDER_PIXELS,
	filler_bbox.width,
	filler_bbox.height);

    // Copy line from bw_img into combined image, inverted to have black
    // text on white background.
    cv::Mat src(filler_si.bw_img, filler_bbox);
    cv::Mat dst(combined_img, target_bbox);
    cv::bitwise_not(src, dst);
  }

  // Copy the line to recover
  {
    const cv::Rect target_bbox(
	NUM_BORDER_PIXELS
	+ filler_bbox.width
	+ NUM_BORDER_PIXELS,
	NUM_BORDER_PIXELS,
	bbox.width,
	bbox.height);
    combined_line_bbox_vec_vec.emplace_back();
    combined_line_bbox_vec_vec[0].emplace_back(target_bbox);

    // Copy line from bw_img into combined image, inverted to have black
    // text on white background.
    cv::Mat src(si.bw_img, bbox);
    cv::Mat dst(combined_img, target_bbox);
    cv::bitwise_not(src, dst);
  }

  return combined_img;
}

void
OCRSubtitles::recover_wipe(
    cv::Mat& combined_img,
    const std::size_t subtitle_i,
    const std::size_t line_i,
    const std::size_t filler_subtitle_i,
    const std::size_t filler_line_i) {

  const SubtitleInfo& si = subtitle_info_vec[subtitle_i];
  const SubtitleInfo& filler_si = subtitle_info_vec[filler_subtitle_i];

  const cv::Rect& bbox = si.bw_line_bbox_vec[line_i];
  const cv::Rect& filler_bbox = filler_si.bw_line_bbox_vec[filler_line_i];

  // Wipe the two instances of filler line in combined_img
  for (std::size_t i = 0; i < 2; i++) {
    int x_pos = NUM_BORDER_PIXELS + i * (
	filler_bbox.width
	+ NUM_BORDER_PIXELS
	+ bbox.width
	+ NUM_BORDER_PIXELS);

    const cv::Rect target_bbox(
	x_pos,
	NUM_BORDER_PIXELS,
	filler_bbox.width,
	filler_bbox.height);

    cv::Mat src(
	filler_bbox.height,
	filler_bbox.width,
	CV_8UC1,
	cv::Scalar(255));
    cv::Mat dst(combined_img, target_bbox);
    src.copyTo(dst);
  }
}

OCRLine
OCRSubtitles::recover_line(
    tesseract::TessBaseAPI& tess_base_api,
    const std::size_t subtitle_i,
    const std::size_t line_i,
    const std::size_t filler_subtitle_i,
    const std::size_t filler_line_i) {

  std::vector<std::vector<cv::Rect>> combined_line_bbox_vec_vec;

  cv::Mat combined_img(recover_prepare(
	subtitle_i,
	line_i,
	filler_subtitle_i,
	filler_line_i,
	combined_line_bbox_vec_vec));

  std::vector<std::vector<OCRLine>> subtitle_line_vec;

  batch_ocr(
      tess_base_api,
      combined_img,
      combined_line_bbox_vec_vec,
      subtitle_i,
      line_i,
      subtitle_line_vec);

  // get the text of filler line
  std::stringstream filler_line_ss;
  subtitle_vec[filler_subtitle_i].write_line(filler_line_ss, filler_line_i);
  std::string filler_line(filler_line_ss.str());
  // cut off newline
  filler_line.pop_back();

  // get the text of the combined line
  std::stringstream line_ss;
  subtitle_line_vec[0][0].write(line_ss);
  std::string line(line_ss.str());
  // cut off newline
  line.pop_back();

  // retrieve the result beteen the filler lines
  if (!line.starts_with(filler_line) || !line.ends_with(filler_line)) {
    std::stringstream ss;
    ss << "subtitle " << (subtitle_i + 1) <<
      ", line " << (line_i + 1) <<
      ": could not recover text (filler line mismatch)";
    throw generic_exception(ss.str());
  }
  line = line.substr(filler_line.size(), line.size() - (2 * filler_line.size()));
  if (line.find_first_not_of(' ') == std::string::npos) {
    std::stringstream ss;
    ss << "subtitle " << (subtitle_i + 1) <<
      ", line " << (line_i + 1) <<
      ": could not recover text (result is empty)";
    throw generic_exception(ss.str());
  }

  std::cerr << "subtitle " << (subtitle_i + 1) <<
      ", line " << (line_i + 1) <<
      ": recovered text: " << line << std::endl;

  // Update the text of the OCRLine
  std::istringstream iss;
  iss.str(line);
  subtitle_line_vec[0][0].read(iss);

  // This wipe is required, because:
  // - The OCRLine has too many OCR bboxes for words and symbols of the fill
  //   line. These wil automatically be detected as invalid and discarded
  //   if the image does not contain any black pixels inside the bboxes.
  // - Contour bboxes are detected over the whole width of the image.
  recover_wipe(
      combined_img,
      subtitle_i,
      line_i,
      filler_subtitle_i,
      filler_line_i);

  return subtitle_line_vec[0][0];
}

void
OCRSubtitles::recover_subtitle(
    tesseract::TessBaseAPI& tess_base_api,
    const std::size_t subtitle_i,
    const bool show) {

  const SubtitleInfo& si = subtitle_info_vec[subtitle_i];

  // TODO if combining with the chosen filler line doesn't work, try a different filler line

  // TODO find a subtitle line to use
  if (subtitle_i > 0 && subtitle_info_vec[0].bw_line_bbox_vec.size()) {
    // use 0
  }
  else {
    // TODO
    return;
  }

  std::vector<OCRLine> line_vec;
  for (std::size_t line_i = 0; line_i < si.bw_line_bbox_vec.size(); line_i++) {

    try {
      line_vec.emplace_back(recover_line(
	  tess_base_api,
	  subtitle_i,
	  line_i,
	  0,
	  0));
    }
    catch (const std::exception& e) {

      std::cerr << e.what() << std::endl;

      std::vector<OCRWord> word_vec;
      std::vector<cv::Rect> word_ocr_bbox_vec;
      std::vector<cv::Rect> symbol_ocr_bbox_vec;
      line_vec.emplace_back(
	  si.subtitle_number,
	  line_i + 1,
	  cv::Mat(),
	  word_vec,
	  cv::Rect(0,0,0,0),
	  word_ocr_bbox_vec,
	  symbol_ocr_bbox_vec);
    }
  }

  subtitle_vec[subtitle_i] = OCRSubtitle(
      si.subtitle_number,
      si.start_pts,
      si.end_pts,
      line_vec);

  if (show) {
    for (std::size_t line_i = 0; line_i < subtitle_vec[subtitle_i].num_lines(); line_i++) {
      std::cout << "Subtitle " << (subtitle_i + 1) << ": ";
      subtitle_vec[subtitle_i].write_line(std::cout, line_i);
    }
  }
}

void
OCRSubtitles::recover(
    tesseract::TessBaseAPI& tess_base_api,
    const bool show) {

  if (subtitle_vec.size() != subtitle_info_vec.size()) {
    throw generic_exception("OCRSubtitles::recover: subtitle_vec size mismatch");
  }

  for (std::size_t i = 0; i < subtitle_info_vec.size(); ++i) {
    const SubtitleInfo& si = subtitle_info_vec[i];
    const OCRSubtitle& subtitle = subtitle_vec[i];

    if (subtitle.num_lines() == 0 && si.bw_line_bbox_vec.size()) {
      recover_subtitle(tess_base_api, i, show);
    }
  }
}

std::ostream&
OCRSubtitles::write(
    std::ostream& os) const {
  for (const auto& it : subtitle_vec) {
    it.write(os);
  }
  return os;
}

void
OCRSubtitles::read(
    std::istream& is) {
  for (auto& it : subtitle_vec) {
    it.read(is);
  }
}

