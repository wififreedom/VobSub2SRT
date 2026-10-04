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

#include "OCRLine.h++"

#include <opencv2/imgcodecs.hpp>

#include "generic_exception.h++"
#include "debug.h++"

#include <codecvt>
#include <cstdint>
#include <locale>
#include <iterator>

OCRLine::OCRLine(
    const std::size_t subtitle_number,
    const std::size_t line_number,
    const cv::Mat& ocr_img,
    const cv::Rect& ocr_bbox,
    std::vector<cv::Rect>& ocr_word_bbox_vec_arg,
    std::vector<cv::Rect>& ocr_symbol_bbox_vec_arg,
    std::vector<OCRWord>& word_vec_arg,
    const cv::Mat& itd_img,
    const cv::Rect& itd_bbox)
    : subtitle_number(subtitle_number),
      line_number(line_number),
      ocr_img(ocr_img),
      ocr_bbox(ocr_bbox),
      ocr_word_bbox_vec(),
      ocr_symbol_bbox_vec(),
      word_vec(),
      itd_img(itd_img),
      itd_bbox(itd_bbox),
      nwv(),
      nsv() {
  ocr_word_bbox_vec.swap(ocr_word_bbox_vec_arg);
  // Tesseract does seem to sort bboxes, but not by x coordinate and then
  // width (maybe by their center?)
  bboxes_sort(ocr_word_bbox_vec);
  ocr_symbol_bbox_vec.swap(ocr_symbol_bbox_vec_arg);
  // Tesseract does seem to sort bboxes, but not by x coordinate and then
  // width (maybe by their center?)
  bboxes_sort(ocr_symbol_bbox_vec);
  word_vec.swap(word_vec_arg);
}

std::size_t
OCRLine::num_ocr_symbols() const {
  std::size_t num = 0;
  for (const auto & word : word_vec) {
    num += word.num_ocr_symbols();
  }
  return num;
}

std::ostream&
OCRLine::write(
    std::ostream& os) const {
  for (bool at_begin = true; const auto& word : word_vec) {
    if (!at_begin) {
      os << ' ';
    }
    word.write(os);
    at_begin = false;
  }
  os << std::endl;
  return os;
}

// does not fully validate:
// - does not validate byte 2,3,4
// - does not check for overlong encodings
// returns empty string on error
static std::string
get_utf8_char(
    const std::string& src,
    const std::size_t pos) {

  std::string dst;

  if (pos < src.size()) {
    unsigned char c0 = src[pos];

    if (c0 <= 0x7f) {
      dst.push_back(src[pos]);
    }
    else if ((c0 & 0xE0) == 0xC0) {
      if ((pos + 1) < src.size()) {
	dst.push_back(src[pos]);
	dst.push_back(src[pos + 1]);
      }
    }
    else if ((c0 & 0xF0) == 0xE0) {
      if ((pos + 2) < src.size()) {
	dst.push_back(src[pos]);
	dst.push_back(src[pos + 1]);
	dst.push_back(src[pos + 2]);
      }
    }
    else if ((c0 & 0xF8) == 0xF0) {
      if ((pos + 3) < src.size()) {
	dst.push_back(src[pos]);
	dst.push_back(src[pos + 1]);
	dst.push_back(src[pos + 2]);
	dst.push_back(src[pos + 3]);
      }
    }
  }

  return dst;
}

void
OCRLine::read(
    std::istream& is) {

  std::string line;

  if (!std::getline(is, line)) {
    std::stringstream ss;
    ss << "subtitle " << subtitle_number <<
      ", line " << line_number <<
      ": could not read modified line after replacements";
    throw generic_exception(ss.str());
  }

  // istream_iterator splits on whitespace sequence
  std::stringstream line_ss(line);
  std::istream_iterator<std::string> begin(line_ss);
  std::istream_iterator<std::string> end;
  const std::vector<std::string> words(begin, end);

  std::vector<OCRWord> new_word_vec;
  std::size_t word_number = 1;
  for (const auto& word_it : words) {
    std::vector<OCRSymbol> symbol_vec;

    for (std::size_t pos = 0; pos < word_it.size(); ) {
      std::string utf8_symbol = get_utf8_char(word_it, pos);
      if (utf8_symbol.size() == 0) {
	std::stringstream ss;
	ss << "subtitle " << subtitle_number <<
	  ", line " << line_number <<
	  ", word " << word_number <<
	  ", byte " << pos + 1 <<
	  "invalid UTF-8 sequence";
	throw generic_exception(ss.str());
      }

      symbol_vec.emplace_back(utf8_symbol);

      pos += utf8_symbol.size();
    }

    new_word_vec.emplace_back(
	subtitle_number,
	line_number,
	word_number,
	std::move(symbol_vec));

    word_number++;
  }

  // compare to see if we need to invalidate word_ocr_bbox_vec.
  if (new_word_vec.size() != word_vec.size()) {
    ocr_word_bbox_vec.clear();
  }

  new_word_vec.swap(word_vec);
}

// Goals:
// - Don't modify the OCR result: symbols, words, lines. Adapt to OCR
//   mistakes by adjusting bounding boxes and excluding bounding boxes
//   from italic detetion instead.
// - Best effort. The quality of the COR result is not 100% perfect. So this
//   algorithm can only try to repair and work around as many issues as
//   possible.
// - Italic detection per word. Not per symbol or per line.
// - Punctuation is often part of OCR words, but should not be considered
//   part of the word for italic detection.
//
// stats is NULL on first call, then after that, stats are built,
// then the method is called again with stats != NULL.
bool
OCRLine::bboxes_assign(
    const std::string& subname,
    const TextStats * const stats) {
  // About symbol_contour_bbox_vec: These are bounding boxes for the symbols
  // in the line, detected with (opencv) contour detection. They are
  // ordered by cv::Rect.x coordinate, ascending.
  // These bounding boxes can be inaccurate:
  // - in case of touching characters: a combined bbox, resulting in
  //   too few bboxes. This is likely to happen sometimes.
  // - for italic characters that do not touch, the bounding boxes
  //   can overlap a little bit: for example for "va", the rightmost
  //   pixel of "v" may be above the first pixel(s) of "a".
  //   This is to be expected because we don't have italic ("slanted" /
  //   "sheared" / "skewed") bounding boxes.
  // - for double quotes vs multiple single quotes: a combined bbox,
  //   resulting in too few bboxes. This is unlikely to happen.

  if (debug || subtitle_number == debug_subtitle_number) {
    cerr_log() << ": bboxes_assign" << (stats ? "" : " (to build stats)") << std::endl;
  }

  std::vector<cv::Rect> symbol_contour_bbox_vec;
  // Tesseract OCR works best with black text on a white background.
  // Contour detection works best with white text on a black background:
  // then the contours are on the inside of the symbols, resulting in
  // accurate bounding boxes.
  bboxes_invert_and_detect(itd_img, itd_bbox, symbol_contour_bbox_vec);
  bboxes_sort_and_combine(symbol_contour_bbox_vec);

  // It's important to improve the quality of the symbol bboxes both for
  // gathering statistics, and for italic detection in general.
  // Problems that can't be fixed by only looking at symbol OCR bboxes and
  // symbol contour bboxes are:
  // - a bbox that spans two touching symbols
  //   - this happens if there is and invalid OCR bbox and a combined countour
  //     bbox because two symbols touch.
  //   - it isn't a problem for statistics because the number of bboxes for
  //     a word won't match the number of symbols of the OCR word, and for that
  //     reason the symbols of the word will not be included in the statistics
  //     (unless there's a second problem that causes one *more* bbox in the
  //     word, in which case it's not detectable, but that happens very
  //     infrequently).
  symbol_bboxes_improve(
    stats ? std::string() : subname,
    ocr_symbol_bbox_vec,
    symbol_contour_bbox_vec,
    nsv);

  word_bboxes_improve(
    stats ? std::string() : subname,
    ocr_word_bbox_vec,
    nsv,
    nwv,
    stats);


  if (stats) {
    // Further improve word bboxes. Potential problems to fix:
    // - word_bboxes_improve() may have removed invalid word bboxes,
    //   and now some symbol bboxes are not bounded by any word bbox.
    // - OCR has produced a word OCR bbox that is not invalid, but
    //   inaccurate and excludes a symbol that is now not part of
    //   any word OCR bbox.
    // - OCR has produced a word OCR bbox that is too narrow and does not
    //   cover all of the symbols of the word completely.
    if (!word_bboxes_bound_all_symbol_bboxes(nwv, nsv) || nwv.size() != word_vec.size()) {
      std::vector<cv::Rect> tmp;

      // Fixing word bboxes is not simple if there are also
      // problems with symbol bboxes.

      // Try to fix by adding symbol bboxes and combining them into words.
      // Vary word spacing for combining symbol bboxes into word bboxes to get
      // at least the required number of word bboxes, or more.
      // This is not exact science: word spacing varies, and OCR may have made
      // a mistake, too.
      std::optional<int> min_word_spacing_opt = stats->word_spacing_min();
      std::optional<float> avg_symbol_spacing_opt = stats->symbol_spacing_avg();
      if (min_word_spacing_opt.has_value() && avg_symbol_spacing_opt.has_value()) {

	const int min_word_spacing_lim = 2 * avg_symbol_spacing_opt.value();
	for (int min_word_spacing = min_word_spacing_opt.value();
	     (tmp.size() < word_vec.size()) && (min_word_spacing >= min_word_spacing_lim);
	     min_word_spacing--) {
	  word_bboxes_fill_gaps_and_combine_based_on_spacing(nwv, nsv, tmp, min_word_spacing);
	}
      }

      if (debug || subtitle_number == debug_subtitle_number) {
	if (debug_ext.size()) {
	  std::stringstream ss;
	  ss << subname << "-" << subtitle_number << "-" << line_number << "-word-supplem-bboxes." << debug_ext;
	  cv::imwrite(ss.str(), bboxes_draw(itd_img, bordered_itd_bbox(), tmp));
	}
	cerr_log() << ": bboxes_assign: word supplem bboxes: ";
	bboxes_stream(std::cerr, tmp) << std::endl;
      }

      // If now we have too many word bboxes, see if we can merge some.
      while (tmp.size() > word_vec.size()) {

	// first evaluate possible merges
	std::vector<std::size_t> possibles;
	for (std::size_t word_i = 0; (word_i + 1) < word_vec.size(); word_i++) {
	  OCRWord& cw = word_vec[word_i];

	  const cv::Rect cr = tmp[word_i];
	  const cv::Rect nr = tmp[word_i + 1];

	  std::vector<cv::Rect> ccands;
	  std::vector<cv::Rect> ncands;

	  bboxes_get_word_candidates(cr, nsv, ccands);
	  bboxes_get_word_candidates(nr, nsv, ncands);

	  if ((cw.num_ocr_symbols() > ccands.size()) &&
	      (cw.num_ocr_symbols() == (ccands.size() + ncands.size()))) {
	    // This is still a guess. It may be that there is a symbol bbox in
	    // cw that covers two symbols.

	    // test if we can repair this word without merge of word bboxes
	    if (!cw.bboxes_assign(itd_img, ccands, stats, true)) {
	      possibles.emplace_back(word_i);
	    }
	    // clear any assigned bboxes after the experiment
	    cw.bboxes_remove();
	  }
	  else {
	    // This is also a guess. There may be overlapping bboxes in cw.
	    // do not consider a possibility.
	  }
	}

	if (possibles.size() == (tmp.size() - word_vec.size())) {
	  if (debug || subtitle_number == debug_subtitle_number) {
	    cerr_log() << ": bboxes_assign: word bboxes: merge index " << possibles[0] <<
	      " and " << (possibles[0] + 1) << std::endl;
	  }
	  // merge
	  tmp[possibles[0]] = tmp[possibles[0]] | tmp[possibles[0] + 1];
	  tmp.erase(tmp.begin() + possibles[0] + 1);

	  if (debug || subtitle_number == debug_subtitle_number) {
	    if (debug_ext.size()) {
	      std::stringstream ss;
	      ss << subname << "-" << subtitle_number << "-" << line_number << "-word-merged-bboxes." << debug_ext;
	      cv::imwrite(ss.str(), bboxes_draw(itd_img, bordered_itd_bbox(), tmp));
	    }
	    cerr_log() << ": bboxes_assign: word merged bboxes: ";
	    bboxes_stream(std::cerr, tmp) << std::endl;
	  }
	}
	else {
	  // If 0 possibles, OCR may have missed a symbol. Seen:
	  // - OCR recognizes "lll" instead of "I'll", so missed a symbol.
	  // Don't second-guess OCR, this code is just for italics detection for now.
	  // TODO: we could warn, though.
	  if (debug || subtitle_number == debug_subtitle_number) {
	    cerr_log() << ": bboxes_assign: word bboxes: FAILED: " <<
	      "merge possibilities inconclusive: " <<
	      possibles.size() << " possibles:";
	    for (const auto& it : possibles) {
	      std::cerr << " " << it;
	    }
	    std::cerr <<  std::endl;
	  }
	  break;
	}
      }

      if (tmp.size() == word_vec.size()) {
	// The number of word bboxes now matches word_vec.size(). Do the
	// numbers of symbols per word match?
	
	// This is the phase where we can try to assign, and still adjust
	// word bboxes by moving the first symbol from the next word,
	// or the last symbol from the previous word.

	// If word i has too few symbols, look at the neighbouring symbol
	// bboxes:
	// - If the previous symbol bbox could match the first symbol of word
	//   i, and the previous symbol bbox could _not_ match the last symbol
	//   of word i-1, move it to word i by adjusting word bboxes.
	// - Same for next symbol bbox, last symbol of word_i, and first
	//   symbol of word i+1.
	// - Can or can't match: always look at average symbol height and width
	//   in OCRWord::bboxes_assign(), and fail.
      }

      nwv.swap(tmp);
    }
  }

  bool success = true;
 
  // If word bboxes are really bad and span multiple words, or OCR has split a
  // word into two (in word_vec), the improved bboxes can end up with fewer or
  // more words. In that case, fall back to the original word_ocr_bbox_vec,
  // if possible.
  //
  // Note: word_ocr_bbox_vec may no longer have the same number of entries as
  // word_vec after regular expresssion replacements have been applied.
  std::vector<cv::Rect> *word_bbox_vec = NULL;
  if (nwv.size() == word_vec.size()) {
    word_bbox_vec = &nwv;
  }
  else {
    if (debug || subtitle_number == debug_subtitle_number) {
      cerr_log() << ": number of words does not match number of word bboxes" <<
	", don't assign symbol bboxes to symbols" << std::endl;
    }
    success = false;
  }

  if (success) {
    // now assign symbol bboxes to symbols
    for (std::size_t word_i = 0; word_i < word_vec.size(); word_i++) {
      OCRWord& word = word_vec[word_i];
      const cv::Rect wr = (*word_bbox_vec)[word_i];
      std::vector<cv::Rect> cands;

      if (debug || subtitle_number == debug_subtitle_number) {
	cerr_log() << ", word " << (word_i + 1) << ": " << word << std::endl;
      }

      bboxes_get_word_candidates(wr, nsv, cands);

      if (debug || subtitle_number == debug_subtitle_number) {
	if (debug_ext.size()) {
	  std::stringstream ss;
	  ss << subname << "-" << subtitle_number << "-" << line_number << "-" << (word_i + 1) << "-word-cands." << debug_ext;
	  cv::imwrite(ss.str(), bboxes_draw(itd_img, bordered_itd_bbox(), cands));
	}
	cerr_log() << ", word " << (word_i + 1) << ": " << word << ": improved symbol bboxes for word: ";
	bboxes_stream(std::cerr, cands) << std::endl;
      }

      if (word.bboxes_assign(itd_img, cands, stats, false)) {
	if (debug || subtitle_number == debug_subtitle_number) {
	  if (debug_ext.size()) {
	    std::stringstream ss;
	    ss << subname << "-" << subtitle_number << "-" << line_number << "-" << (word_i + 1) << "-symbol-assigned-bboxes." << debug_ext;
	    cv::imwrite(ss.str(), itd_word_symbol_bboxes_draw(word_i, 128));
	  }
	  cerr_log() << ", word " << (word_i + 1) << ": " << word << ": assigned symbol bboxes for word: ";
	  bboxes_stream(std::cerr, word.itd_symbol_bboxes_get()) << std::endl;
	}
      } else {
	success = false;
      }
    }
  }

  if (success) {
    return true;
  }

  if (stats) {
    if (debug || subtitle_number == debug_subtitle_number) {
      cerr_log() << ": bboxes_assign: FAILED" << std::endl;
    }
  }
  return false;
}

void
OCRLine::build_stats(
    TextStats& stats) const {
  for (auto& it : word_vec) {
    it.build_stats(itd_img, stats);
  }

  for (std::size_t i = 1; i < nwv.size(); i++) {
    const int space_pixels = nwv[i].x - (nwv[i - 1].x + nwv[i - 1].width);
    // Word bboxes are not reliable if they overlap.
    if (space_pixels > 0)
      stats.word_spacing_add(space_pixels);
  }
}

void
OCRLine::bboxes_remove() {
  for (auto& it : word_vec) {
    it.bboxes_remove();
  }
}

void
OCRLine::assign_confidence(
    const TextStats& stats) {
  for (auto& word: word_vec) {
    word.assign_confidence(itd_img, stats);
  }
}

void
OCRLine::propagate_word_confidence() {
  // Determine word parts and derive italic confidence for word parts from
  // symbols. The italic confidence may be inconclusive for some parts.
  for (auto& it : word_vec) {
    it.word_parts_determine();
  }

  // Propagate italic_confidence for word parts that have uncertain
  // italic confidence, where possible.
  bool have_uncertain = false;
  bool made_change = false;
  do {
    have_uncertain = false;
    made_change = false;

    for (std::size_t i = 0; i < word_vec.size(); i++) {
      OCRWord& word = word_vec[i];

      const float prev_confidence = (i > 0) ? word_vec[i - 1].end_confidence() : DEFAULT_CONFIDENCE;
      const float next_confidence = ((i + 1) < word_vec.size()) ? word_vec[i + 1].begin_confidence() : DEFAULT_CONFIDENCE;

      word.word_parts_confidence_propagate(
	  prev_confidence,
	  next_confidence,
	  have_uncertain,
	  made_change);
    }
  } while (have_uncertain && made_change);

  if (debug || subtitle_number == debug_subtitle_number) {
    std::cerr << "subtitle " << subtitle_number << ", line " << line_number << ": " << std::endl;
    dump(std::cerr);
  }
}

void
OCRLine::write_srt(std::ostream& os) const {
  bool entire_line_is_italic = true;
  bool in_italic = false;
  for (std::size_t i = 0; i < word_vec.size(); i++) {
    word_vec[i].write_srt(
	os,
	((i + 1) == word_vec.size()) ? NULL : &(word_vec[i + 1]),
       	in_italic,
       	entire_line_is_italic);
  }
}

std::size_t
OCRLine::num_chars_for_duration() const {
  std::size_t num = 0;
  for (const auto& word : word_vec) {
    // +1 for space or newline
    num += word.num_ocr_symbols() + 1;
  }
  return num;
}

void
OCRLine::dump(std::ostream& os) const {
  for (const auto& word : word_vec) {
    word.dump(os);
  }
}

std::ostream&
OCRLine::cerr_log() const {
  return std::cerr << "subtitle " << subtitle_number <<
    ", line " << line_number;
}

cv::Rect
OCRLine::bordered_itd_bbox() const {
  return cv::Rect(
      itd_bbox.x - NUM_BORDER_PIXELS,
      itd_bbox.y - NUM_BORDER_PIXELS,
      itd_bbox.width + 2 * NUM_BORDER_PIXELS,
      itd_bbox.height + 2 * NUM_BORDER_PIXELS);
}

// Note that a double quote has a column of whitespace in the middle.
// Specify check_all_columns = true only if certain that the bbox is not for a
// symbol that may have a white column, such as a double-quote.
bool
OCRLine::bbox_is_invalid(
    const cv::Mat& img,
    const cv::Rect& bbox,
    const bool check_all_columns) const {

  // bbox has width <= 0 or height <= 0: bad
  if (bbox.width <= 0 || bbox.height <= 0) {
    return true;
  }

  if (check_all_columns)
  {
    // bbox has at least 1 column of only white pixels: bad
    for (int i = 0; i < bbox.width; i++) {
      int j = 0;
      for (; j < bbox.height; j++) {
	if (img.at<uchar>(bbox.y + j, bbox.x + i) != ((uchar) 255)) {
	  break;
	}
      }
      if (j == bbox.height) {
	// white-only column found, bad bbox
	return true;
      }
    }
  }
  else {
    // bbox has a left column with only white pixels: bad
    int i = 0;
    for (i = 0; i < bbox.height; i++) {
      if (img.at<uchar>(bbox.y + i, bbox.x) != ((uchar) 255)) {
	break;
      }
    }
    if (i == bbox.height) {
      return true;
    }

    // bbox has a right column with only white pixels: bad
    for (i = 0; i < bbox.height; i++) {
      if (img.at<uchar>(bbox.y + i, bbox.x + bbox.width - 1) != ((uchar) 255)) {
	break;
      }
    }
    if (i == bbox.height) {
      return true;
    }
  }

  // bbox has a top row with only white pixels: bad
  {
    int i = 0;
    for (; i < bbox.width; i++) {
      if (img.at<uchar>(bbox.y, bbox.x + i) != ((uchar) 255)) {
	break;
      }
    }
    if (i == bbox.width) {
      return true;
    }
  }

  // OCR bbox has a bottom row with only white pixels: bad
  {
    int i = 0;
    for (; i < bbox.width; i++) {
      if (img.at<uchar>(bbox.y + bbox.height - 1, bbox.x + i) != ((uchar) 255)) {
	break;
      }
    }
    if (i == bbox.width) {
      return true;
    }
  }

  return false;
}

void
OCRLine::symbol_bboxes_remove_invalid(
    const cv::Mat& img,
    const cv::Rect& line_bbox,
    const std::vector<cv::Rect>& src /* symbol bboxes */,
    std::vector<cv::Rect>& dst /* symbol bboxes */) {

  dst.clear();

  const int dq_y_limit = line_bbox.y + (line_bbox.height / 2);

  for (const auto& bbox : src) {
    const bool maybe_double_quote = ((bbox.y + bbox.height) < dq_y_limit);

    if (!bbox_is_invalid(img, bbox, !maybe_double_quote)) {
      dst.emplace_back(bbox);
    }
  }
}

void
OCRLine::symbol_bboxes_ocr_to_itd(
    const std::vector<cv::Rect>& src /* ocr symbol bboxes */,
    std::vector<cv::Rect>& dst /* itd symbol bboxes */) {

  dst.clear();

  for (const auto& src_bbox : src) {
    dst.emplace_back(src_bbox);
    bbox_shrink(itd_img, dst.back());
  }
}

void
OCRLine::symbol_bboxes_fill_gaps(
    const std::vector<cv::Rect>& src,
    const std::vector<cv::Rect>& src2,
    std::vector<cv::Rect>& dst) {

  // Strategy:
  // - Add all src bboxes to dst. they may overlap.
  // - Add all src2 bboxes that cover some range that is not covered by any
  //   src bbox.
  // - Don't deal with resulting overlap in this function, that's for other
  //   functions to deal with later.
  // - At return elements of dst are in ascending order of cv::Rect.x.

  dst.clear();

  std::size_t src2_i = 0;
  std::size_t src_i = 0;
  int src_max_x = -1;
  while (src2_i < src2.size()) {
    // add and skip all src bboxes that come before this src2 bbox.
    while ((src_i < src.size()) && (src[src_i].x + src[src_i].width) <= src2[src2_i].x) {
      dst.emplace_back(src[src_i]);
      src_max_x = std::max(src_max_x, src[src_i].x + src[src_i].width);
      src_i++;
    }

    // skip all src bboxes until a gap occurs
    while ((src_i < src.size()) && (src[src_i].x <= src_max_x)) {
      dst.emplace_back(src[src_i]);
      src_max_x = std::max(src_max_x, src[src_i].x + src[src_i].width);
      src_i++;
    }

    if (src_i < src.size()) {
      // gap from src_max_x (inclusive) to src[src_i].x (exclusive)
      if ((src2[src2_i].x + src2[src2_i].width) <= src_max_x) {
	// src2 comes completely before gap, so skip
	src2_i++;
      }
      else if (src2[src2_i].x >= src[src_i].x) {
	// src2 comes completely after gap, so we can't fill the gap
	// (space between symbols)
	dst.emplace_back(src[src_i]);
	src_max_x = std::max(src_max_x, src[src_i].x + src[src_i].width);
	src_i++;
      }
      else {
	// src2 covers (part of) gap
	dst.emplace_back(src2[src2_i]);
	src2_i++;
      }
    }
    else {
      // gap from src_max_x (inclusive) to infinity (no more src bboxes)
      if ((src2[src2_i].x + src2[src2_i].width) <= src_max_x) {
	// src2 comes completely before gap, so skip
	src2_i++;
      } else {
	dst.emplace_back(src2[src2_i]);
	src2_i++;
      }
    }
  }
  while (src_i < src.size()) {
    dst.emplace_back(src[src_i]);
    src_i++;
  }
}

void
OCRLine::symbol_bboxes_replace_combined(
  const std::vector<cv::Rect>& src,
  const std::vector<cv::Rect>& src2,
  std::vector<cv::Rect>& dst) {

  dst.clear();

  // If possible, replace one OCR bbox from src by two contour bboxes from
  // src2 of which the union bounds the OCR bbox in x direction.
  // This improves situations with a combined OCR bbox for, for example:
  // - italic "y.",
  // - non-italic "xt,", "fo"
  // and possibly for more situations where the combined bbox was not deemed
  // invalid by bbox_is_invalid(), because there is no vertical line of white
  // pixels in between the symbols.
  // Contour bboxes are correct in these situations, because the pixels of the
  // symbols not touch.
  std::size_t src_i = 0;
  std::size_t src2_i = 0;
  while (src_i < src.size()) {
    const cv::Rect& r = src[src_i];

    while ((src2_i < src2.size()) && (src2[src2_i].x < r.x)) {
      src2_i++;
    }

    if ((src2_i < src2.size()) &&
       	(src2[src2_i].x == r.x) && 
	((src2_i + 1) < src2.size()) &&
	((src2[src2_i + 1].x + src2[src2_i + 1].width) ==
	 (r.x + r.width))) {
      dst.emplace_back(src2[src2_i]);
      src2_i++;
      dst.emplace_back(src2[src2_i]);
      src2_i++;
      src_i++;
    } else {
      dst.emplace_back(r);
      src_i++;
    }
  }
}

// TODO should we guarantee that all bounding box vectors are ordered by:
// - X coordinate (already the case)
// - and then also by WIDTH (not the case yet)

// This is not 100% guaranteed to be an improvement
// Take theoretical italic "mrv", where:
// - there is a good OCR bounding box for "m" and "r", but the OCR bounding box
//   for "v" was invalid
// - the contour bboxes are for "m" and "rv", because r and v touch
// This will result in src having improved bounding boxes "m", "r", and "rv".
// If "m" and "r" overlap a little due to italic, and "m" and "rv" as well for
// the same reason, and "r" and "rv" overlap, this will remove "r", which
// is a better bounding box than "rv". In that case, we'll have to split
// a combined bbox later.
void
OCRLine::symbol_bboxes_remove_inaccurate_overlapping(
    const std::vector<cv::Rect>& src,
    std::vector<cv::Rect>& dst) {

  dst.clear();

  // Sometimes tesseract creates a wildly inaccurate OCR bbox for an
  // OCR symbol S. Typically still in the x range of the OCR word,
  // but that's not guaranteed.
  // - Tesseract then sorts the OCR bboxes by x position, inserting
  //   the inaccurate OCR bbox for OCR symbol S in the wrong place.
  //   Note: Tesseract does not always sort by x perfectly. Seen:
  //   23:(373,1702)-(386,1722), 24:(372,1702)-(404,1722)
  //   So how does Tesseract do it, exactly? Does it sort by the center
  //   of the bbox maybe?
  // - Tesseract then returns this inaccurate OCR bbox for the wrong
  //   symbol, and the OCR bboxes for other OCR symbols can be shifted.
  // - At a different position in the ordered OCR bboxes, an OCR bbox
  //   is missing, or an OCR bbox covers two OCR symbols.
  //
  // If the current OCR bbox is such an inaccurate OCR bbox, there
  // can be four different cases:
  // a. The current OCR bbox overlaps both previous and next OCR bbox.
  // b. The current OCR bbox overlaps only the next OCR bbox.
  // c. The current OCR bbox overlaps only the previous OCR bbox.
  // d. The current OCR bbox does not overlap any other OCR bbox.
  //
  // Case a has been handled by remove_invalid_bboxes for instances where
  // there was whitespace in between prev and next bbox.
  // Now we look at where previous and next bbox touch or in case
  // of italic, slightly overlap.
  for (std::size_t i = 0; i < src.size(); i++) {
    const cv::Rect& r = src[i];

    if ((i > 0) && ((i + 1) < src.size())) {
      const cv::Rect& p = src[i - 1];
      const cv::Rect& n = src[i + 1];
      if (/* current overlaps prev */
	  (r.x < (p.x + p.width)) &&
	  /* current overlaps next */
	  ((r.x + r.width) > n.x) &&
	  /* prev touches or overlaps next */
	  ((p.x + p.width) >= n.x)) {
	// do not add to dst
	continue;
      }
    }
    dst.emplace_back(r);
  }
}

void
OCRLine::symbol_bboxes_remove_overlapped_by_1(
    const std::vector<cv::Rect>& src,
    std::vector<cv::Rect>& dst) {

  dst.clear();

  for (std::size_t i = 0; i < src.size(); i++) {
    const cv::Rect& r = src[i];
    if (dst.size()) {
      const cv::Rect& p = dst.back();
      if ((r.x + r.width) <= (p.x + p.width)) {
	// skip r
	continue;
      }
    }

    if ((i + 1) < src.size()) {
      const cv::Rect& n = src[i + 1];
      if (r.x == n.x && (r.x + r.width) <= (n.x + n.width)) {
	// skip r
	continue;
      }
    }

    dst.emplace_back(r);
  }
}

void
OCRLine::symbol_bboxes_improve(
    const std::string& subname,
    const std::vector<cv::Rect>& src, // ocr bboxes
    const std::vector<cv::Rect>& src2, // contour bboxes
    std::vector<cv::Rect>& dst) {

  dst.clear();

  if (debug || subtitle_number == debug_subtitle_number) {
    if (subname.size() && debug_ext.size()) {
      std::stringstream ss;
      ss << subname << "-" << subtitle_number << "-" << line_number << "-symbol-ocr-bboxes." << debug_ext;
      cv::imwrite(ss.str(), bboxes_draw(ocr_img, bordered_itd_bbox(), src));
    }
    cerr_log() << ": symbol ocr      bboxes: ";
    bboxes_stream(std::cerr, src) << std::endl;

    if (subname.size() && debug_ext.size()) {
      std::stringstream ss;
      ss << subname << "-" << subtitle_number << "-" << line_number << "-symbol-contour-bboxes." << debug_ext;
      cv::imwrite(ss.str(), bboxes_draw(itd_img, bordered_itd_bbox(), src2));
    }
    cerr_log() << ": symbol contour  bboxes: ";
    bboxes_stream(std::cerr, src2) << std::endl;
  }

  std::vector<cv::Rect> tmp, tmp2;

  symbol_bboxes_remove_invalid(ocr_img, ocr_bbox, src, tmp);

  symbol_bboxes_ocr_to_itd(tmp, tmp2);

  // itd_img can have fewer pixels, so some characters that touch in ocr_img
  // may not touch in itd_img. This means that it may be possible to detect
  // more invalid OCR bboxes after conversion to itd bboxes.
  symbol_bboxes_remove_invalid(itd_img, itd_bbox, tmp2, tmp);

  symbol_bboxes_fill_gaps(tmp, src2, tmp2);

  symbol_bboxes_replace_combined(tmp2, src2, tmp);

  symbol_bboxes_remove_inaccurate_overlapping(tmp, tmp2);

  // Strategy: remove all bboxes that are completely overlapped by one other
  // bbox, then repair any combined bboxes.
  symbol_bboxes_remove_overlapped_by_1(tmp2, dst);

  // After these steps, all symbols should be fully covered.
  //
  // There can still be overlapping bboxes:
  // 1 Seen: for "tw" that touch each other: a good OCR bbox for the "t", 
  //   completely overlapped by a combined contour bbox for "tw".
  // 2 Seen: for "b" at the begin of a word: there is a bad OCR bbox that
  //   bounds the first half of the width of "b", completely overlapped by a
  //   correct contour bbox for "b". Also seen for "h", "F", "B", "I" at
  //   the begin of a word.
  // 3 Maybe other cases?
  if (debug || subtitle_number == debug_subtitle_number) {
    if (subname.size() && debug_ext.size()) {
      std::stringstream ss;
      ss << subname << "-" << subtitle_number << "-" << line_number << "-symbol-improved-bboxes." << debug_ext;
      cv::imwrite(ss.str(), bboxes_draw(itd_img, bordered_itd_bbox(), dst));
    }
    cerr_log() << ": symbol improved bboxes: ";
    bboxes_stream(std::cerr, dst) << std::endl;
  }
}

void
OCRLine::word_bboxes_remove_overlapping(
    const std::vector<cv::Rect>& src,
    std::vector<cv::Rect>& dst) {

  dst.clear();

  // TODO the next one could also overlap the one after it
  for (std::size_t i = 0; i < src.size(); i++) {
    if ((i + 1) < src.size() && ((src[i].x + src[i].width) > src[i + 1].x)) {
      // remove this one and the next one
      i++;
    }
    dst.emplace_back(src[i]);
  }
}

void
OCRLine::word_bboxes_remove_invalid(
    const cv::Mat &img,
    const std::vector<cv::Rect>& src /* word bboxes */,
    std::vector<cv::Rect>& dst /* word bboxes */,
    const TextStats* const /* stats */) {

  dst.clear();

  for (const auto& bbox : src) {
    if (!bbox_is_invalid(img, bbox, false)) {
      dst.emplace_back(bbox);
    }
  }
}

void
OCRLine::word_bboxes_ocr_to_itd(
    const std::vector<cv::Rect>& src /* ocr word bboxes */,
    std::vector<cv::Rect>& dst /* itd word bboxes */) {

  dst.clear();

  for (const auto& src_bbox : src) {
    dst.emplace_back(src_bbox);
    bbox_shrink(itd_img, dst.back());
  }
}

void
OCRLine::word_bboxes_remove_too_much_spacing(
    const std::vector<cv::Rect>& src, // word bboxes
    std::vector<cv::Rect>& dst,
    const TextStats& stats) {

  dst.clear();

  // Strategy: remove word bboxes that may overlap multiple words,
  // because they have suspiciously many sequential white columns.
  //
  // To be recombined according to word_vec later.
  std::optional<int> min_word_spacing_opt = stats.word_spacing_min();
  std::optional<float> avg_symbol_spacing_opt = stats.symbol_spacing_avg();
  if (!min_word_spacing_opt.has_value() || !avg_symbol_spacing_opt.has_value()) {
    dst = src;
    return;
  }

  const int max_spacing = std::max(min_word_spacing_opt.value(), (int)(2 * avg_symbol_spacing_opt.value()));

  for (const auto& bbox : src) {
    const int max_seq_white_columns = bbox_max_seq_white_columns(itd_img, bbox);
    if (max_seq_white_columns <= max_spacing) {
      dst.emplace_back(bbox);
    }
  }
}

void
OCRLine::word_bboxes_improve(
    const std::string& subname,
    const std::vector<cv::Rect>& src,
    const std::vector<cv::Rect>& /* symbol_src */,
    std::vector<cv::Rect>& dst,
    const TextStats * const stats) {

  if (debug || subtitle_number == debug_subtitle_number) {
    if (subname.size() && debug_ext.size()) {
      std::stringstream ss;
      ss << subname << "-" << subtitle_number << "-" << line_number << "-word-ocr-bboxes." << debug_ext;
      cv::imwrite(ss.str(), bboxes_draw(itd_img, bordered_itd_bbox(), src));
    }
    cerr_log() << ": word ocr      bboxes: ";
    bboxes_stream(std::cerr, src) << std::endl;
  }

  std::vector<cv::Rect> tmp, tmp2;

  // Remove overlapping before removing invalid bboxes.
  // In case of an invalid bbox, there may also be an overlapping bbox that
  // bounds two words. We can only detect that overlapping bbox before removing
  // the invalid bboxes.
  word_bboxes_remove_overlapping(src, tmp);

  word_bboxes_remove_invalid(ocr_img, tmp, tmp2, stats);

  word_bboxes_ocr_to_itd(tmp2, tmp);

  if (!stats) {
    // itd_img can have fewer pixels, so some characters that touch in ocr_img
    // may not touch in itd_img. This means that it may be possible to detect
    // more invalid OCR bboxes after conversion to itd bboxes.
    // This is less likely with word bboxes than with symbol bboxes, but check
    // anyway.
    word_bboxes_remove_invalid(itd_img, tmp, dst, stats);
  }
  else {
    // itd_img can have fewer pixels, so some characters that touch in ocr_img
    // may not touch in itd_img. This means that it may be possible to detect
    // more invalid OCR bboxes after conversion to itd bboxes.
    // This is less likely with word bboxes than with symbol bboxes, but check
    // anyway.
    word_bboxes_remove_invalid(itd_img, tmp, tmp2, stats);

    word_bboxes_remove_too_much_spacing(tmp2, dst, *stats);
  }

  // TODO also remove other types of inaccurate bboxes.
  //      for example: if any of the contour bboxes partially or completely overlap a word bbox,
  //      the word bbox can't be trusted.

  if (debug || subtitle_number == debug_subtitle_number) {
    if (subname.size() && debug_ext.size()) {
      std::stringstream ss;
      ss << subname << "-" << subtitle_number << "-" << line_number << "-word-improved-bboxes." << debug_ext;
      cv::imwrite(ss.str(), bboxes_draw(itd_img, bordered_itd_bbox(), dst));
    }
    cerr_log() << ": word improved bboxes: ";
    bboxes_stream(std::cerr, dst) << std::endl;
  }
}

bool
OCRLine::word_bboxes_bound_all_symbol_bboxes(
    const std::vector<cv::Rect>& src, // word bboxes
    const std::vector<cv::Rect>& src2) { // symbol bboxes

  std::size_t src2_i = 0;
  std::size_t src_i = 0;
  int src_max_x = -1;
  while (src2_i < src2.size()) {
    // skip all src bboxes that come before this src2 bbox.
    while ((src_i < src.size()) && (src[src_i].x + src[src_i].width) <= src2[src2_i].x) {
      src_max_x = std::max(src_max_x, src[src_i].x + src[src_i].width);
      src_i++;
    }

    // skip all src bboxes until a gap occurs
    while ((src_i < src.size()) && (src[src_i].x <= src_max_x)) {
      src_max_x = std::max(src_max_x, src[src_i].x + src[src_i].width);
      src_i++;
    }

    if (src_i < src.size()) {
      // gap from src_max_x (inclusive) to src[src_i].x (exclusive)
      if ((src2[src2_i].x + src2[src2_i].width) <= src_max_x) {
	// src2 comes completely before gap, so skip
	src2_i++;
      }
      else if (src2[src2_i].x >= src[src_i].x) {
	// src2 comes completely after gap, so we can't fill the gap
	// (space between symbols)
	src_max_x = std::max(src_max_x, src[src_i].x + src[src_i].width);
	src_i++;
      }
      else {
	// src2 covers (part of) gap
	return false;
      }
    }
    else {
      // gap from src_max_x (inclusive) to infinity (no more src bboxes)
      if ((src2[src2_i].x + src2[src2_i].width) <= src_max_x) {
	// src2 comes completely before gap, so skip
	src2_i++;
      } else {
	return false;
      }
    }
  }

  return true;
}

void
OCRLine::word_bboxes_fill_gaps_and_combine_based_on_spacing(
    const std::vector<cv::Rect>& src, // word bboxes
    const std::vector<cv::Rect>& src2, // symbol bboxes
    std::vector<cv::Rect>& dst,
    const int min_word_spacing) {

  dst.clear();

  std::vector<cv::Rect> tmp;

  // Fill the gaps with symbol bboxes
  symbol_bboxes_fill_gaps(src, src2, tmp);

  // Combine added symbol bboxes into words.
  // Notes:
  // - An inaccurate too narrow src word bbox may have been completely
  //   overlapped by a symbol bbox from src2. TODO
  std::size_t tmp_i = 0;
  while (tmp_i < tmp.size()) {
    cv::Rect r = tmp[tmp_i];

    cv::Rect new_word(r);
    tmp_i++;
    while (tmp_i < tmp.size()) {
      cv::Rect rr = tmp[tmp_i];
      if (rr.x >= (new_word.x + new_word.width + min_word_spacing)) {
	break;
      }
      new_word = new_word | rr; // union
      tmp_i++;
    }
    dst.emplace_back(new_word);
  }
}

void
OCRLine::bboxes_get_word_candidates(
  const cv::Rect& word,
  const std::vector<cv::Rect>& src,
  std::vector<cv::Rect>& dst) {

  dst.clear();

  for (const auto& r : src) {
    if ((r.x >= word.x) && ((r.x + r.width) <= (word.x + word.width))) {
      dst.emplace_back(r);
    }
    else if ((r.x < (word.x + word.width)) && ((r.x + r.width) > word.x)) {
      if (debug) {
	cerr_log() << ": bboxes_get_word_candidates: partial overlap: w: ";
	bbox_stream(std::cerr, word);
	std::cerr << ", s: ";
	bbox_stream(std::cerr, r);
	std::cerr << std::endl;
      }
      // inaccurate word bounding box or inaccurate symbol bounding box.
      // do not add symbol bounding box.
    }
    else if (r.x > (word.x + word.width)) {
      break;
    }
  }
}

cv::Mat
OCRLine::itd_word_symbol_bboxes_draw(
  std::size_t const word_index,
  unsigned char grayscale_color) const {

  if (word_index > word_vec.size()) {
    throw generic_exception("symbol_bboxes_draw: word index out of range");
  }

  const cv::Rect itd_img_bbox = bordered_itd_bbox();
  cv::Mat result_itd_img(itd_img_bbox.height, itd_img_bbox.width, itd_img.type());
  cv::Mat(itd_img, itd_img_bbox).copyTo(result_itd_img);

  word_vec[word_index].itd_symbol_bboxes_draw(result_itd_img, itd_img_bbox, grayscale_color);

  return result_itd_img;
}

cv::Mat
OCRLine::itd_symbol_bboxes_draw(
  unsigned char grayscale_color) const {

  const cv::Rect itd_img_bbox = bordered_itd_bbox();
  cv::Mat result_itd_img(itd_img_bbox.height, itd_img_bbox.width, itd_img.type());
  cv::Mat(itd_img, itd_img_bbox).copyTo(result_itd_img);

  for (const auto& word : word_vec) {
    word.itd_symbol_bboxes_draw(result_itd_img, itd_img_bbox, grayscale_color);
  }

  return result_itd_img;
}

