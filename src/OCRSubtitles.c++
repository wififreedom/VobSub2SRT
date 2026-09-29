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
      bw_line_bbox_vec(),
      combined_line_bbox_vec() {

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
      for (std::size_t i = batch_i; i < batch_end_i; i++) {
	std::cout << "Subtitle " << (i + 1) << ": ";
	subtitle_vec[i].write(std::cout);
      }
    }
  }
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
    const std::size_t batch_begin_i,
    const std::size_t batch_end_i) {

  std::string msg;

  // now do ocr.
  try {
    batch_ocr(
	tess_base_api,
	batch_prepare(batch_begin_i, batch_end_i),
	batch_begin_i,
	batch_end_i);

    return;
  }
  catch (const generic_exception& e) {
    msg = e.what();
  }

  // Could be out of sync due to OCR not recognizing any text in an entire line.
  // Try to repair by splitting the batch, if necessary repeatedly.
  const std::size_t batch_size = batch_end_i - batch_begin_i;
  if (batch_size > 1) {
    const std::size_t batch_split_i = batch_begin_i + (batch_size / 2);

    do_ocr(tess_base_api, batch_begin_i, batch_split_i);

    do_ocr(tess_base_api, batch_split_i, batch_end_i);
  }
  else {
    // The problem is in this one subtitle.
    // Maybe placing text in front and after can make OCR recognize something.

    // TODO this code is proven to work in at least 1 case, but needs to be cleaned up
    // - split off the code to create the combined_img
    // - just replace the subtitle completely, with a nice small img instead of the combined_img.
    // - if combining with line 1 doesn't work, try a different line
    // - handle the case where there are two lines in the subtitle

    if (batch_begin_i > 0 &&
	subtitle_info_vec[batch_begin_i].bw_line_bbox_vec.size() == 1) {
      if (subtitle_info_vec[0].bw_line_bbox_vec.size()) {

	subtitle_info_vec[batch_begin_i].combined_line_bbox_vec.clear();

	const cv::Rect& line1_bbox = subtitle_info_vec[0].bw_line_bbox_vec[0];
	const cv::Rect& current_line_bbox = subtitle_info_vec[batch_begin_i].bw_line_bbox_vec[0];
	const int max_height = std::max(line1_bbox.height, current_line_bbox.height);

	// TODO if possible, align the text of line1 and current line vertically

	const int combined_img_width =
	  NUM_BORDER_PIXELS /* left border */
	  + line1_bbox.width
	  + NUM_BORDER_PIXELS /* horizontal spacing between lines of text */
	  + current_line_bbox.width
	  + NUM_BORDER_PIXELS /* horizontal spacing between lines of text */
	  + line1_bbox.width
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

	// Place first line of first subtitle in front and after
	// line of current subtitle.
	for (std::size_t i = 0; i < 2; i++) {
	  int x_pos = NUM_BORDER_PIXELS + i * (
	      line1_bbox.width
	      + NUM_BORDER_PIXELS
	      + current_line_bbox.width
	      + NUM_BORDER_PIXELS);

	  const cv::Rect target_bbox(
	      x_pos,
	      NUM_BORDER_PIXELS,
	      line1_bbox.width,
	      line1_bbox.height);

	  // Copy line from bw_img into combined image, inverted to have black
	  // text on white background.
	  cv::Mat src(subtitle_info_vec[0].bw_img, line1_bbox);
	  cv::Mat dst(combined_img, target_bbox);
	  cv::bitwise_not(src, dst);
	}

	{
	  const cv::Rect target_bbox(
	      NUM_BORDER_PIXELS
	      + line1_bbox.width
	      + NUM_BORDER_PIXELS,
	      NUM_BORDER_PIXELS,
	      current_line_bbox.width,
	      current_line_bbox.height);
	  subtitle_info_vec[batch_begin_i].combined_line_bbox_vec.emplace_back(target_bbox);

	  // Copy line from bw_img into combined image, inverted to have black
	  // text on white background.
	  cv::Mat src(subtitle_info_vec[batch_begin_i].bw_img, current_line_bbox);
	  cv::Mat dst(combined_img, target_bbox);
	  cv::bitwise_not(src, dst);
	}

	try {
	  batch_ocr(
	      tess_base_api,
	      combined_img,
	      batch_begin_i,
	      batch_end_i);

	  // Now we have text, but too much.

	  // get the text of the first line of the first subtitle
	  std::stringstream subtitle_1_ss;
	  subtitle_vec[0].write(subtitle_1_ss);
	  std::string subtitle_1(subtitle_1_ss.str());
	  // cut off newline and any second line of first subtitle
	  const std::size_t pos = subtitle_1.find('\n');
	  subtitle_1.resize(pos);

	  std::stringstream current_line_ss;
	  subtitle_vec[batch_begin_i].write(current_line_ss);
	  std::string current_line(current_line_ss.str());
	  // cut off newline
	  current_line.pop_back();
	  if (!current_line.starts_with(subtitle_1) ||
	      !current_line.ends_with(subtitle_1)) {
	    // TODO better message
	    throw generic_exception("could not recover");
	  }
	  current_line = current_line.substr(subtitle_1.size(), current_line.size() - (2 * subtitle_1.size()));
	  // TODO better message
	  std::cerr << "recovered: " << current_line << std::endl;

	  // Update the text of the OCRSubtitle
	  std::istringstream iss;
	  iss.str(current_line);
	  subtitle_vec[batch_begin_i].read(iss);

	  // Wipe the two instances of first line in combined_img
	  for (std::size_t i = 0; i < 2; i++) {
	    int x_pos = NUM_BORDER_PIXELS + i * (
		line1_bbox.width
		+ NUM_BORDER_PIXELS
		+ current_line_bbox.width
		+ NUM_BORDER_PIXELS);

	    const cv::Rect target_bbox(
		x_pos,
		NUM_BORDER_PIXELS,
		line1_bbox.width,
		line1_bbox.height);

	    cv::Mat src(
		line1_bbox.height,
		line1_bbox.width,
		CV_8UC1,
		cv::Scalar(255));
	    cv::Mat dst(combined_img, target_bbox);
	    src.copyTo(dst);
	  }

	  return;
	}
	catch (const generic_exception& e) {
	  std::cerr << e.what() << std::endl;
	}
      }
    }

    std::cerr << msg << std::endl;
  }
}

cv::Mat
OCRSubtitles::batch_prepare(
    const std::size_t batch_begin_i,
    const std::size_t batch_end_i) {

  // clear all data set by a previous call to batch_prepare
  for (std::size_t si_i = batch_begin_i; si_i < batch_end_i; si_i++) {
    subtitle_info_vec[si_i].combined_line_bbox_vec.clear();
  }

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
      for (const auto& lb_it : subtitle_info_vec[si_i].bw_line_bbox_vec) {
	const cv::Rect target_bbox(
	    NUM_BORDER_PIXELS,
	    NUM_BORDER_PIXELS + line_i * (line_max_height + NUM_BORDER_PIXELS),
	    lb_it.width,
	    lb_it.height);
	subtitle_info_vec[si_i].combined_line_bbox_vec.emplace_back(target_bbox);

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
    const std::size_t batch_i,
    const std::size_t batch_end_i) {

  // clear all data set by a previous call to batch_ocr
  while (subtitle_vec.size() > batch_i) {
    subtitle_vec.pop_back();
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

  std::size_t si_i = batch_i; /* subtitle_info_vec index */
  std::vector<OCRLine> line_vec;
  std::vector<OCRWord> word_vec;
  std::vector<cv::Rect> word_bbox_vec;
  std::vector<cv::Rect> symbol_bbox_vec;

  do {
    bool first = true;

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
	// This can only happen if the image is empty. In all other cases
	// ri->Next returns false and the loop exits.
	std::unique_ptr<char[]> symbol(ri->GetUTF8Text(tesseract::RIL_SYMBOL));
	if (!symbol) {
	  std::stringstream ss;
	  ss << "WARNING: subtitle " << (si_i + 1) <<
	    ", line " << (line_vec.size() + 1) <<
	    ": ocr: image has no text (--batch-size with a larger value may fix this)";
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
	  ss << "subtitle " << (si_i + 1) <<
	    ", line " << (line_vec.size() + 1) <<
	    ": ocr: could not get symbol bounding box";
	  throw generic_exception(ss.str());
	}
	symbol_bbox_vec.emplace_back(
	  cv::Rect(left, top, right - left, bottom - top));

      } while (!ri->IsAtFinalElement(tesseract::RIL_WORD, tesseract::RIL_SYMBOL));

      word_vec.emplace_back(
	  si_i + 1,
	  line_vec.size() + 1,
	  word_vec.size() + 1,
	  std::move(symbol_vec));

      int left = 0, top = 0, right = 0, bottom = 0;
      if (!ri->BoundingBox(tesseract::RIL_WORD,
	    &left, &top, &right, &bottom)) {
	std::stringstream ss;
	ss << "subtitle " << (si_i + 1) <<
	  ", line " << (line_vec.size() + 1) <<
	  ": ocr: could not get word bounding box";
	throw generic_exception(ss.str());
      }
      word_bbox_vec.emplace_back(
	cv::Rect(left, top, right - left, bottom - top));
    } while (!ri->IsAtFinalElement(tesseract::RIL_TEXTLINE, tesseract::RIL_SYMBOL));

    // skip subtitles with blank images
    while (si_i < batch_end_i && subtitle_info_vec[si_i].combined_line_bbox_vec.size() == 0) {
      si_i++;
    }

    if (si_i < batch_end_i) {
      const SubtitleInfo& si = subtitle_info_vec[si_i];

      line_vec.emplace_back(
	  si_i + 1,
	  line_vec.size() + 1,
	  word_vec,
	  si.combined_line_bbox_vec[line_vec.size()],
	  word_bbox_vec,
	  symbol_bbox_vec);

      if (line_vec.size() == si.combined_line_bbox_vec.size()) {
	subtitle_vec.emplace_back(
	    si.subtitle_number,
	    si.start_pts,
	    si.end_pts,
	    combined_img,
	    line_vec);

	si_i++;
      }
    }
    else {
      std::stringstream ss;
      ss << "subtitle " << (si_i + 1) <<
	", line " << (line_vec.size() + 1) <<
	": ocr: more lines than expected for batch";
      throw generic_exception(ss.str());
    }

  } while (ri->Next(tesseract::RIL_SYMBOL));

  if (si_i != batch_end_i) {
    std::stringstream ss;
    ss << "subtitle " << (si_i + 1) <<
      ", line " << (line_vec.size() + 1) <<
      ": ocr: fewer lines than expected for batch";
    throw generic_exception(ss.str());
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

