/*
 *  VobSub2SRT is a simple command line program to convert .idx/.sub subtitles
 *  into .srt text subtitles by using OCR (tesseract). See README.
 *
 *  Copyright (C) 2010-2016 Rüdiger Sonderfeld <ruediger@c-plusplus.de>
 *  Copyright (C) 2026 Christopher Ogloff <chris.ogloff@gmail.com>
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

// Tesseract OCR
#include <tesseract/baseapi.h>

// Builtins/standard libs
#include <fstream>
#include <format>
#include <regex>

// Language and option handling.
#include "langcodes.h++"
#include "cmd_options.h++"
#include "version.h++"

// Italics detection with help of opencv
#include "debug.h++"
#include "OCRSubtitles.h++"
#include "Replacements.h++"
#include "VobSub.h++"

#include <opencv2/imgcodecs.hpp>

void
process_sub(
    const std::string& sub_file,
    const std::string& ifo_file,
    const std::string& dump_ext,
    const bool show,
    const int verbosity,
    const bool list_languages,
    const int sub_index,
    const std::string& sub_lang,
    const int y_threshold,
    const std::string& tess_data_dir,
    const std::string& tess_lang_user,
    const std::string& tess_blacklist,
    const std::string& tess_dpi,
    const std::size_t ocr_batch_size,
    const Replacements& replacements,
    const bool detect_italic,
    const unsigned base_duration,
    const unsigned chars_per_sec
    )
{
  std::string subname = sub_file;

  // cut off .idx or .sub, if present.
  if (subname.size() >= 4) {
    const std::string ext = subname.substr(subname.size() - 4, 4);
    if (ext == ".idx" || ext == ".sub") {
      subname.resize(subname.size() - 4);
    }
  }

  OCRSubtitles subtitles(subname);

  // Set Y threshold
  if (y_threshold != 16) {
    std::cerr << "Using Y palette threshold: " << y_threshold << "\n";
  }

  // Open the sub/idx subtitles
  VobSub vobsub;
  if (verbosity) {
    std::cerr << "Reading '" << subname << ".idx'" << std::endl;
  }
  vobsub.open(subname, ifo_file, y_threshold);

  // list languages and return
  if (list_languages) {
    std::cout << "'" << subname << "': languages:\n";
    for (size_t i = 0; i < vobsub_get_indexes_count(vobsub.vob()); ++i) {
      const char* const id = vobsub_get_id(vobsub.vob(), i);
      std::cout << i << ": " << (id ? id : "(no id)") << '\n';
    }
    return;
  }

  // Tesseract: set default OCR language to eng, unless overridden by user
  // or inferred from selected subtitle track.
  const char* tess_lang = tess_lang_user.empty() ? "eng" : tess_lang_user.c_str();
  if (!sub_lang.empty()) {
    if(vobsub_set_from_lang(vobsub.vob(), sub_lang.c_str()) < 0) {
      std::cerr << "No matching language for '" << sub_lang << "' found! (Trying to use default)\n";
    }
    else if (tess_lang_user.empty()) {
      const char* const lang3 = iso639_1_to_639_3(sub_lang.c_str());
      if (lang3) {
	tess_lang = lang3;
      }
    }
  } else {
    if (sub_index >= 0) {
      if (static_cast<unsigned>(sub_index) >= vobsub_get_indexes_count(vobsub.vob())) {
	std::cerr << "Index argument out of range: " << sub_index << " " <<
	  "(" << vobsub_get_indexes_count(vobsub.vob()) << ")\n";
	return;
      }
      vobsub_id = sub_index;
    }
    
    if (vobsub_id >= 0) {
      const char* const lang1 = vobsub_get_id(vobsub.vob(), vobsub_id);
      if (lang1 && tess_lang_user.empty()) {
	const char* const lang3 = iso639_1_to_639_3(lang1);
	if (lang3) {
	  tess_lang = lang3;
	}
      }
    }
  }
  
  // Tesseract: announce language data dir for verbosity. This needs a bit of work.
  if (verbosity && !tess_data_dir.empty()) {
    std::cerr << "Using Tesseract data directory: '" << tess_data_dir << "'\n";
  }

  // Tesseract: init
  tesseract::TessBaseAPI tess_base_api;
  if (tess_base_api.Init(
	(tess_data_dir.empty() ? NULL : tess_data_dir.c_str()),
       	tess_lang,
       	tesseract::OEM_LSTM_ONLY) == -1) {
    throw generic_exception("Failed to initialize Tesseract (OCR)");
  }
  
  // Tesseract: blacklist if not empty
  if (!tess_blacklist.empty()) {
    tess_base_api.SetVariable("tessedit_char_blacklist", tess_blacklist.c_str());
  }
  
  // Tesseract: run multithreaded
  // TODO test: does this produce better, same, or worse results
  tess_base_api.SetVariable("tessedit_parallelize", "1");

  // Tesseract: set DPI to 72, or user-specified with --dpi
  // TODO test: do different values produce better results
  tess_base_api.SetVariable("user_defined_dpi", tess_dpi.c_str());

  // Tesseract: attempt to improve whitespace detection
  tess_base_api.SetVariable("tessedit_use_reject_spaces", "0");
  tess_base_api.SetVariable("tosp_improve_thresh", "1");

  // Open srt output file
  std::ofstream srt_ofs;
  try {
    srt_ofs.exceptions(std::ios_base::failbit | std::ios_base::badbit);
    srt_ofs.open(subname + ".srt",
	std::ios_base::out | std::ios_base::trunc);
  }
  catch (...) {
    throw generic_exception("Could not open .srt file for writing");
  }

  // Read subtitles and convert
  void *packet;
  int timestamp; // pts100
  int len;
  unsigned last_start_pts = 0;
  unsigned last_end_pts = 0;
  unsigned sub_counter = 1;

  if (verbosity) {
    std::cerr << "Reading subtitle start/end times and images from '" << subname << ".sub'" << std::endl;
  }

  while ((len = vobsub_get_next_packet(vobsub.vob(), &packet, &timestamp)) > 0) {
    if (timestamp >= 0) {
      spudec_assemble(vobsub.spu(), reinterpret_cast<unsigned char*>(packet), len, timestamp);
      if (!spudec_heartbeat(vobsub.spu(), timestamp)) {
	// spudec_assemble is still assembling packet from its fragments
	continue;
      }

      unsigned char const *sp_image;
      unsigned sp_width, sp_height, sp_stride;
      size_t sp_image_size; // should be at least height * stride
      unsigned start_pts = 0, end_pts = 0;

      // Get the image data to know dimensions and timing info
      spudec_get_data(vobsub.spu(), &sp_image, &sp_image_size, &sp_width, &sp_height, &sp_stride, &start_pts, &end_pts);

      if (sp_width == 0 || sp_height == 0) {
	std::cerr << sub_counter << ": start " << start_pts <<
	  ", image width: " << sp_width << ", height: " << sp_height <<
	  ": skipping" << std::endl;
	continue;
      }

      if (verbosity and static_cast<unsigned>(timestamp) != start_pts) {
	std::cerr << sub_counter << ": time stamp from .idx (" << timestamp
		  << ") doesn't match time stamp from .sub ("
		  << start_pts << ")\n";
      }

      // deal with subtitles that have the same start time
      if (start_pts == last_start_pts && last_end_pts != UINT_MAX) {
	start_pts = last_end_pts;
      }
      last_start_pts = start_pts;
      last_end_pts = end_pts;

      // dump subtitle image to file for option --dump-images
      if (dump_ext.size()) {
	std::stringstream ss;
	ss << subname << "-" << std::format("{:04}", sub_counter) << "." << dump_ext;
	cv::Mat sp_img(
	    static_cast<int>(sp_height),
	    static_cast<int>(sp_width),
	    CV_8UC1,
	    (void *) sp_image,
	    static_cast<std::size_t>(sp_stride));
	cv::imwrite(ss.str(), sp_img);
      }

      const bool have_text = subtitles.append(
	  sub_counter,
	  start_pts,
	  end_pts,
	  sp_image,
	  sp_width,
	  sp_height,
	  sp_stride);

      if (have_text) {
	sub_counter++;
      }
    }
    else {
      std::cerr << "timestamp < 0" << std::endl;
    }
  }

  if (verbosity) {
    std::cerr << "Performing OCR" << std::endl;
  }
  subtitles.do_ocr(tess_base_api, ocr_batch_size, show);

  if (verbosity) {
    std::cerr << "Performing replacements (if any)" << std::endl;
  }
  subtitles.correct_ocr(replacements);

  if (detect_italic) {
    if (verbosity) {
      std::cerr << "Performing italic detection" << std::endl;
    }
    subtitles.detect_italic();
  }

  if (verbosity) {
    std::cerr << "Writing subtitles to '" << subname << ".srt'" << std::endl;
  }
  subtitles.write_srt(srt_ofs, base_duration, chars_per_sec);

  std::cerr << "Wrote Subtitles to '" << subname << ".srt'\n";
}

int
main2(int argc, char **argv) {
  bool show = false;
  std::string dump_ext;
  int verbosity = -1;
  std::string ifo_file;
  bool list_languages = false;
  int sub_index = -1;
  std::string sub_lang;
  std::string tess_lang_user;
  std::string tess_blacklist;
  std::string tess_data_dir;
  std::string tess_dpi = "72";
  int y_threshold = 16;
  std::size_t ocr_batch_size = 20;
  std::vector<std::string> replacements_file_name_vec;
  bool detect_italic = false;
  unsigned base_duration = 0;
  unsigned chars_per_sec = 19;
  std::vector<std::string> subname_vec;

  {
    cmd_options opts;
    unsigned debug_st_num = 0;
    unsigned batch_size = 20;
    opts.
      add_option("show", show, "Show subtitles being written.").
      add_option("dump-images", dump_ext, "Save subtitle image files with specified extension (<subname>-<number>.ext) (pgm, png, ...).").
      add_option("debug", debug, "Write all kinds of debug info to standard error.").
      add_option("debug-number", debug_st_num, "Write all kinds of debug info to standard error, for a specific subtitle number.").
      add_option("debug-images", debug_ext, "Also write debug images for options --debug and --debug-number (pgm, png, ...).").
      add_option("verbose", verbosity, "Verbosity, a value of 1 or 2. Also applies to decoder.").
      add_option("ifo", ifo_file, "Name of the IFO file. Default: tries to open <subname>.ifo(case insensitive).\n\t\t\t\tIFO file is optional but may fix empty palette issues!").
      add_option("index", sub_index, "Subtitle index to select. Incompatible with option --lang.", 'i').
      add_option("lang", sub_lang, "Subtitle language to select. Incomaptible with option --index.", 'l').
      add_option("langlist", list_languages, "List subtitle languages present in the .idx/.sub files and exit").
      add_option("tesseract-lang", tess_lang_user, "Desired Tesseract language (e.g. eng, deu, fra, esp, eng+fra)\n\t\t\t\t(Default: autodetect)").
      add_option("tesseract-data", tess_data_dir, "Path to Tesseract data (e.g. you have tessdata_best and wish to\n\t\t\t\tuse it. Default: autodetect)").
      add_option("dpi", tess_dpi, "Set DPI for Tesseract OCR. Default: 72.").
      add_option("tess_blacklist", tess_blacklist, "Character blacklist to improve the OCR (e.g. \"|\\/`_~<>\").").
      add_option("y-threshold", y_threshold, "Y (luminance) threshold below which colors treated as black. Default: 16.").
      add_option("ocr-batch-size", batch_size, "Perform OCR on combined images. Can fix empty or inaccurate OCR results.\n\t\t\t\tDefault: 20.").
      add_option("replacements", replacements_file_name_vec, "Immediately after OCR, apply replacements defined in the specified file(s).\n\t\t\t\tOption can be specified multiple times.").
      add_option("detect-italic", detect_italic, "Detect italic. Add <i> and </i> to the output where applicable.").
      add_option("base-duration", base_duration, "Max subtitle display duration (msec) = \n\t\t\t\t  base_duration + 1000 * subtitle_length_in_chars / chars_per_sec\n\t\t\t\tDefault: 0 = disable, recommended: 1500.").
      add_option("chars-per-sec", chars_per_sec, "See --base-duration. Default: 19, recommended: 15..20.");

    opts.add_unnameds(subname_vec, "subname", "Name of one of the .idx/.sub files. Ending .idx/.sub optional.\n\t\t\t\tMultiple names of .idx/.sub files can be specified.");
    std::cerr << "VobSub2SRT version " << version << '\n';
    if(!opts.parse_cmd(argc, argv)) {
      return 0;
    }

    debug_subtitle_number = static_cast<std::size_t>(debug_st_num);

    if (verbosity < 0) {
      verbosity = 0;
    }

    ocr_batch_size = static_cast<std::size_t>(batch_size);

    if (!sub_lang.empty() && sub_index >= 0) {
      std::cerr << "Specifiying both --lang and --index not supported.\n";
      return 1;
    }

  }

  if (verbosity > 0) {
    verbose = verbosity; // mplayer verbose level
  }
  
  // Read the replacements file first, to immediately report syntax errors in
  // the file, instead of after OCR has finished.
  Replacements replacements;
  for (const auto& it : replacements_file_name_vec) {
    if (verbosity) {
      std::cerr << "Reading replacements from '" << it << "'" << std::endl;
    }
    replacements.read(it);
  }

  for (const auto& sub_file : subname_vec) {
    process_sub(
	sub_file,
	ifo_file,
	dump_ext,
	show,
	verbosity,
	list_languages,
	sub_index,
	sub_lang,
	y_threshold,
	tess_data_dir,
	tess_lang_user,
	tess_blacklist,
	tess_dpi,
	ocr_batch_size,
	replacements,
	detect_italic,
	base_duration,
	chars_per_sec);
  }
  return 0;
}

int
main(int argc, char **argv) {
  try {
    return main2(argc, argv);
  }
  catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
  }
  return 1;
}
