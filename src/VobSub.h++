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

#include <string>

#include "generic_exception.h++"

#include "mp_msg.h"
#include "vobsub.h"
#include "spudec.h"

typedef void* vob_t;
typedef void* spu_t;

#ifndef VOBSUB_HXX
#define VOBSUB_HXX

class VobSub {

public:
  VobSub();

  ~VobSub();

  void open(
      const std::string& sub_file_name,
      const std::string& ifo_file_name,
      const int y_threshold);

  // returns false on end of stream
  // else, fills in the public data members below.
  bool next();

  void close();

  vob_t vob();

private:
  void
  reset_public();

  void
  reset_private();

public:
  // Image display start time.
  // To convert to milliseconds, divide by 90.
  unsigned start_pts;

  // Image display end time.
  // To convert to milliseconds, divide by 90.
  unsigned end_pts;

  // Image data, property of spudec module, do not modify or deallocate.
  // One byte per pixel.
  // Data pointed to by sp_image persists until the next call to
  // VobSub::next() or VobSub.close().
  const unsigned char* sp_image;

  // Image width, in pixels = bytes.
  unsigned sp_width;

  // Image height, in pixels = bytes.
  unsigned sp_height;

  // sp_stride is sp_width, but rounded up to align the image rows to some power of 2.
  // Pixel 0 of row X is at sp_image[X * sp_stride].
  unsigned sp_stride;

  // Image size, in bytes. Should be at least sp_height * sp_stride.
  size_t sp_image_size;

private:
  spu_t priv_spu;
  vob_t priv_vob;

  unsigned last_start_pts = 0;
  unsigned last_end_pts = 0;
};

#endif // VOBSUB_HXX

