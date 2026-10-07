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

#include "Replacements.h++"

#include "generic_exception.h++"
#include "debug.h++"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/stat.h>

Replacements::Replacement::Replacement(
    const std::regex_constants::syntax_option_type syntax_options,
    const std::regex_constants::match_flag_type match_flags,
    const std::string& match_pattern,
    const std::string& replacement_pattern)
    : priv_syntax_options(syntax_options),
      priv_match_flags(match_flags),
      priv_match_pattern(match_pattern),
      priv_replacement_pattern(replacement_pattern) {
}

void
Replacements::Replacement::replace(
    const std::string& src,
    std::string& dst) const {

  try {
    const std::regex regex(priv_match_pattern, priv_syntax_options);

    dst = std::regex_replace(
	src,
	regex,
	priv_replacement_pattern,
	priv_match_flags);
  }
  catch (const std::exception& e) {
    std::stringstream ss;
    ss << "match pattern: >" << priv_match_pattern << "<" <<
        ", replacement pattern: >" << priv_replacement_pattern << "<" <<
	": " << e.what();
    throw generic_exception(ss.str());
  }
}

static bool
read_line(
    std::ifstream& ifs,
    std::string& line,
    int& line_number,
    bool skip_empty_line) {
  do {
    line_number++;
    if (!std::getline(ifs, line)) {
      return false;
    }

  } while ((line.size() && line[0] == '#') ||
      (skip_empty_line && std::strspn(line.c_str(), " \t") == line.size()));

  return true;
}

void
Replacements::read_file(
    const std::string& file_name,
    const bool verbose)
{
  // Open replacements input file
  std::ifstream ifs;
  try {
    ifs.exceptions(std::ios_base::failbit | std::ios_base::badbit);
    ifs.open(file_name,
	std::ios_base::in | std::ios_base::binary);
    ifs.exceptions(std::ios_base::badbit);
  } catch (...) {
    throw generic_exception("Could not open replacements file '" + file_name + "' for reading");
  }

  if (verbose) {
    std::cerr << "Reading replacements from '" << file_name << "'" << std::endl;
  }

  int line_number = 0;

  std::string header;
  if (!read_line(ifs, header, line_number, true)) {
    return;
  }
  if (header != "ecmascript") {
    std::stringstream ss;
    ss << "'" << file_name << "': line " << line_number << ": file should have a line 'ecmascript' before replacment patterns.";
    throw generic_exception(ss.str());
  }

  while (true) {
    std::string tmp;
    std::string match_pattern;
    std::string replacement_pattern;

    // empty line(s) or match pattern
    if (!read_line(ifs, match_pattern, line_number, true)) {
      return;
    }

    // replacement pattern
    if (!read_line(ifs, replacement_pattern, line_number, false)) {
      std::stringstream ss;
      ss << "'" << file_name << "': line " << line_number << ": missing replacement pattern.";
      throw generic_exception(ss.str());
    }

    // add
    repl_vec.emplace_back(
	(std::regex_constants::ECMAScript |
	  std::regex_constants::multiline |
	  std::regex_constants::optimize),
	std::regex_constants::format_sed,
	match_pattern,
	replacement_pattern);

    // Test the replacements: if one of the regular expressions contains a
    // syntax error, such as mismatched '(' and ')', report it immediately.
    try {
      std::string src, dst;
      repl_vec.back().replace(src, dst);
    } catch (const std::exception& e) {
      std::stringstream ss;
      ss << "'" << file_name << "': lines " << (line_number - 1) <<
	" and " << line_number << ": " << e.what();
      throw generic_exception(ss.str());
    }

    if (debug) {
      std::cerr << "'" << file_name << "': lines " << (line_number - 1) <<
	" and " << line_number << ": " <<
	"added match pattern: >" << match_pattern << "<" <<
	", replacement pattern: >" << replacement_pattern << "<" << std::endl;
    }

    // empty line or end of file
    if (!read_line(ifs, tmp, line_number, false)) {
      break;
    }
    if (std::strspn(tmp.c_str(), " \t") != tmp.size()) {
      std::stringstream ss;
      ss << "'" << file_name << "': line " << line_number << ": missing empty line after replacement pattern.";
      throw generic_exception(ss.str());
    }
  }
}

void
Replacements::read(
    const std::string& path,
    const bool verbose) {

  struct stat sb;
  std::vector<std::string> file_path_vec;

  if (stat(path.c_str(), &sb) == -1) {
    std::stringstream ss;
    ss << "'" << path << "': could not read replacements: " << strerror(errno);
    throw generic_exception(ss.str());
  }
  if (sb.st_mode & S_IFDIR) {
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
      const std::filesystem::path entry_path = entry.path();
      const std::string& entry_name = entry_path.filename();

      if (entry_name.starts_with("replacements_") &&
	  entry_name.ends_with(".txt") &&
	  (stat(entry_path.string().c_str(), &sb) == 0) &&
	  (sb.st_mode & S_IFREG)) {
	file_path_vec.emplace_back(entry_path.string());
      }
    }
  }
  else if (sb.st_mode & S_IFREG) {
    file_path_vec.emplace_back(path);
  }

  // sort by byte value, to get predictable results
  std::sort(file_path_vec.begin(), file_path_vec.end());

  for (const auto& file_path : file_path_vec) {
    read_file(file_path, verbose);
  }
}

std::string
Replacements::replace(
    const std::string& src) const {

  std::string tmp(src);
  std::string tmp2;
  for (const auto& it : repl_vec) {
    it.replace(tmp, tmp2);
    std::swap(tmp, tmp2);
  }
  return tmp;
}

