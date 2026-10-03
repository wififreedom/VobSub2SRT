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

#include <ostream>
#include <vector>

#include <opencv2/core.hpp>

#include "bbox.h++"
#include "stats.h++"
#include "OCRSymbol.h++"

#ifndef OCR_WORD_HXX
#define OCR_WORD_HXX

// An OCRword is a sequence of symbols not separated by whitespace, as produced
// by OCR. It can include punctuation. It can in practice include more than one
// word, for example OCRword [he's] includes punctuation and two words.
// Other examples: ["hello], [dear"], [out!], [will,], [two...], [..three].
class OCRWord {

private:
  struct OCRWordPart {
    OCRWordPart(
	const std::size_t begin_index,
	const std::size_t end_index,
	const bool is_punct_at_begin,
	const bool is_punct_at_end,
	const float italic_confidence);

    std::ostream&
    dump(
	std::ostream& os) const;

    std::size_t begin_index; /* inclusive */
    std::size_t end_index; /* exclusive */
    bool is_punct_at_begin;
    bool is_punct_at_end;
    float italic_confidence;
  };

public:
  OCRWord(
      const std::size_t subtitle_number,
      const std::size_t line_number,
      const std::size_t word_number,
      std::vector<OCRSymbol>&& symbol_vec)
      : subtitle_number(subtitle_number),
        line_number(line_number),
	word_number(word_number),
	symbol_vec(std::move(symbol_vec)),
 	part_vec() {
  }
      
  std::size_t
  num_ocr_symbols() const {
    return symbol_vec.size();
  }

  bool
  bboxes_assign(
      const cv::Mat& img,
      const std::vector<cv::Rect>& src,
      const TextStats* const stats,
      const bool test_only);

  void
  build_stats(
      const cv::Mat& img,
      TextStats& stats) const;

  void
  bboxes_remove();

  void
  assign_confidence(
      const cv::Mat& img,
      const TextStats& stats);

  void
  word_parts_determine();

  float
  begin_confidence() const;

  float
  end_confidence() const;

  void
  word_parts_confidence_propagate(
      const float prev_confidence,
      const float next_confidence,
      bool& have_uncertain,
      bool& made_change);

  std::ostream&
  write(
      std::ostream& os) const;

  std::ostream&
  write_srt(
      std::ostream& os,
      const OCRWord* next,
      bool& in_italic,
      bool& entire_line_is_italic) const;

  void
  dump(
      std::ostream& os) const;

  void
  itd_symbol_bboxes_draw(
    const cv::Mat& img,
    const cv::Rect& line_bbox,
    unsigned char grayscale_color) const;

private:
  bool
  bboxes_assign(
      const std::vector<cv::Rect>& src,
      const TextStats* stats,
      const bool test_only);

  bool
  bboxes_assign_repair_too_few_bboxes(
      const cv::Mat& img,
      const std::vector<cv::Rect>& src,
      const TextStats& stats,
      const bool test_only);

  float
  italic_confidence(
      const std::size_t begin_index /* inclusive */,
      const std::size_t end_index /* exclusive */) const;

  std::ostream&
  part_write(
      std::ostream& os,
      const OCRWordPart& part) const;

private:
  // The number of the subtitle in the OCRSubtitles, starts at 1.
  std::size_t subtitle_number;

  // The number of the line in the OCRSubtitle, starts at 1.
  std::size_t line_number;

  // The number of the word in the OCRLine, starts at 1.
  std::size_t word_number;

  // The symbols of the OCRWord. 
  std::vector<OCRSymbol> symbol_vec;

  // OCRWord parts:
  // - if present, a part for punctuation at begin
  // - a part for each single quote
  // - a part for each sequence of alphabetical symbols
  // - if present, a part for punctuation at end
  std::vector<OCRWordPart> part_vec;
};

inline std::ostream&
operator<<(std::ostream& os, const OCRWord& word) {
  return word.write(os);
}

#endif // OCR_WORD_HXX

