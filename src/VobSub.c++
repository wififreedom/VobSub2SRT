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

#include "VobSub.h++"

#include "generic_exception.h++"

// MPlayer code
#include "mp_msg.h"
#include "vobsub.h"
#include "spudec.h"

#include <climits>
#include <iostream>
#include <sstream>

int VobSub::count = 0;

VobSub::VobSub()
    : priv_vob(NULL),
      priv_spu(NULL) {
  if (count > 0) {
    throw generic_exception("VobSub: only one instance at a time");
  }

  mp_msg_init();
  reset_public();
  reset_private();

  count++;
}

VobSub::~VobSub() {
  close();
  mp_msg_uninit();
  count--;
}

void
VobSub::verbose(
    const int level) {
  ::verbose = level;
}

void
VobSub::open(
    const std::string& sub_file_name,
    const std::string& ifo_file_name,
    const int y_threshold) {

  if (sub_file_name.empty()) {
    throw generic_exception("VobSub::open: empty file name");
  }

  if (priv_vob) {
    throw generic_exception("VobSub::open: already open");
  }

  priv_vob = vobsub_open(
      sub_file_name.c_str(),
      ifo_file_name.empty() ? NULL : ifo_file_name.c_str(),
      1,
      y_threshold,
      &priv_spu);
  if (!priv_vob || vobsub_get_indexes_count(priv_vob) == 0) {
    close();
    throw generic_exception(
	std::string("Couldn't open VobSub files '") +
       	sub_file_name + ".idx and " +
       	sub_file_name + ".sub'");
  }

  reset_public();
  reset_private();
}

std::vector<std::string>
VobSub::languages() const {
  std::vector<std::string> result;

  const std::size_t count = vobsub_get_indexes_count(priv_vob);
  for (std::size_t i = 0; i < count; i++) {
    const char* language = vobsub_get_id(priv_vob, i);
    result.emplace_back(language ? language : "");
  }

  return result;
}

std::string
VobSub::set_subtitle_index(
    const unsigned index) {

  const unsigned index_count = vobsub_get_indexes_count(priv_vob);

  if (index < index_count) {
    vobsub_id = index;
  }
  else {
    std::stringstream ss;
    ss << "subtitle index " << index << " out of range, maximum value: " <<
      (index_count - 1);
    throw generic_exception(ss.str());
  }

  stream_has_been_set = true;

  const char* const lang1 = vobsub_get_id(priv_vob, index);
  return lang1 ? lang1 : "";
}

bool
VobSub::set_subtitle_index_by_language(
    const std::string& language) {

  if (vobsub_set_from_lang(priv_vob, language.c_str()) < 0) {
    return false;
  }

  stream_has_been_set = true;
  return true;
}

bool
VobSub::next() {

  if (!priv_vob || !priv_spu) {
    throw generic_exception("VobSub::next: not open");
  }

  reset_public();

  while (true) {
    unsigned char *packet = NULL;
    int packet_pts100 = 0;

    // Get next mpeg packet
    int packet_size = vobsub_get_next_packet(priv_vob, (void **) &packet, &packet_pts100);
    if (packet_size == -1) {
      // end of stream
      return false;
    }
    if (packet_pts100 == ((int) UINT_MAX)) {
      // bad packet, skip
      continue;
    }

    // The mpeg packet contains a vobsub frame, and spudec_assemble assembles
    // frames into a vobsub packet, decodes the vobsub packet once complete,
    // and places it in a queue.
    spudec_assemble(priv_spu, packet, packet_size, packet_pts100);

    // Function spudec_heartbeat dequeues all vobsub packets from the queue
    // that have a pts100 value <= the provided pt100 value, in this case
    // packet_pts100.
    // It has been modified for vobsub2srt to return the number of packets it
    // dequeued.
    int num_dequeued = spudec_heartbeat(priv_spu, packet_pts100);
    if (num_dequeued == 0) {
      // no vobsub packet fully assembled yet
      continue;
    }

    if (num_dequeued > 1) {
      // Somehow a vobsub packet was skipped, this is unexpected.
      std::cerr << "VobSub Packet skipped???" << std::endl;
    }

    // vobsub packet fully assembled
    // get the data of the assembled vobsub packet
    spudec_get_data(priv_spu, &sp_image, &sp_image_size, &sp_width, &sp_height,
	&sp_stride, &start_pts, &end_pts);

    // try to deal with subtitles that have the same start time
    if (start_pts == last_start_pts && last_end_pts != UINT_MAX) {
      start_pts = last_end_pts;
    }
    last_start_pts = start_pts;
    last_end_pts = end_pts;

    break;
  }

  return true;
}

void
VobSub::close() {
  if (priv_vob) {
    vobsub_close(priv_vob);
    priv_vob = NULL;
  }
  if (priv_spu) {
    spudec_free(priv_spu);
    priv_spu = NULL;
  }
  reset_public();
  reset_private();

  // restore default vobsub id (mplayer uses global variable)
  vobsub_id = 0;
}

void
VobSub::reset_public() {
  start_pts = 0;
  end_pts = 0;
  sp_image = NULL;
  sp_width = 0;
  sp_height = 0;
  sp_stride = 0;
  sp_image_size = 0;
}

void
VobSub::reset_private() {
  stream_has_been_set = false;
  last_start_pts = 0;
  last_end_pts = 0;
}

