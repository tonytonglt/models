# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# This file was copied from project [calendar_day_1][Towards-Realtime-MOT]
"""Multiple objects tracking."""
from collections import deque

import numpy as np

from src.kalman_filter import KalmanFilter
from src.log import logger
from src.utils import non_max_suppression
from src.utils import scale_coords
from tracker import matching
from tracker.basetrack import BaseTrack, TrackState


class TrackS(BaseTrack):
    """
    Compute stracks.
    """
    def __init__(self, tlwh, score, temp_feat, buffer_size=30):
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

    def update_features(self, feat):
        """
        Update values.
        """
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        """
        Compute math distribution.
        """
        mean_state = self.mean.copy()
        if self.state != TrackState.tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks, kalman_filter):
        """
        Compute multi math distribution.
        """
        if stracks:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = kalman_filter.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """
        Start a new tracklet.
        """
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.tracked
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """
        Reactivate new tracks.
        """
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean,
            self.covariance,
            self.tlwh_to_xyah(new_track.tlwh),
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track.

        Args:
            new_track (TrackS): New track frame.
            frame_id (int): Number of current frame.
            update_feature (bool): Update or not.
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    def tlwh(self):
        """
        Get current position in bounding box format
        (top left x, top left y, width, height).
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """
        Convert bounding box to format
        (min x, min y, max x, max y), i.e., (top left, bottom right).
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """
        Convert bounding box to format
        (center x, center y, aspect ratio, height),
        where the aspect ratio is width / height.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        """
        Convert tlwh format to xyah.
        """
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        """
        Convert tlbr format to tlwh.
        """
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        """
        Convert tlwh format to tlbr.
        """
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return f'OT_{self.track_id}_({self.start_frame}-{self.end_frame})'


class JDETracker:
    """
    Compute track per frame and apply tracking.
    """
    def __init__(self, opt, net, frame_rate=30):
        self.opt = opt

        self.model = net
        if opt.infer310:
            logger.info('Inference for 310')
        else:
            logger.info('Inference for: %s', opt.ckpt_url)

        self.tracked_stracks = []  # type: list[TrackS]
        self.lost_stracks = []  # type: list[TrackS]
        self.removed_stracks = []  # type: list[TrackS]

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size

        self.kalman_filter = KalmanFilter()

    def tracking(
            self,
            activated_stracks,
            refind_stracks,
            lost_stracks,
            removed_stracks,
            unconfirmed,
            tracked_stracks,
            detections,
        ):
        """
        Apply tracking strategy.
        """
        # Step 2: First association, with embedding.
        # Combining currently tracked_stracks and lost_stracks
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with kalman filter
        TrackS.multi_predict(strack_pool, self.kalman_filter)

        # Compute distances of the detection with the tracks in strack_pool.
        dists = matching.embedding_distance(strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)
        # The matches is the array for corresponding matches of the detection with the corresponding strack_pool.

        for itracked, idet in matches:
            # itracked is the id of the track and idet is the detection
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.tracked:
                # If the track is active, add the detection to the track
                track.update(detections[idet], self.frame_id)
                activated_stracks.append(track)
            else:
                # Detection from a track which is not active, hence put the track in refind_stracks list
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Step 3: Second association, with IOU
        detections = [detections[i] for i in u_detection]
        # detections is now a list of the unmatched detections
        r_tracked_stracks = []  # This is container for stracks which were tracked till the
        # previous frame but no detection was found for it in the current frame
        for i in u_track:
            if strack_pool[i].state == TrackState.tracked:
                r_tracked_stracks.append(strack_pool[i])
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)
        # matches is the list of detections which matched with corresponding tracks by IOU distance method
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        # Same process done for some unmatched detections, but now considering IOU_distance as measure

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.lost:
                track.mark_lost()
                lost_stracks.append(track)
        # If no detections are obtained for tracks (u_track),
        # the tracks are added to lost_tracks list and are marked lost.

        # Deal with unconfirmed tracks, usually tracks with only one beginning frame
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])

        # The tracks which are yet not matched
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # After all these confirmation steps, if a new detection is found, it is initialized for a new track
        # Step 4: Init new stracks
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # Step 5: Update state
        # If the tracks are lost for more frames than the threshold number, the tracks are removed.
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # Update the self.tracked_stracks and self.lost_stracks using the updates in this step.
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)


    def update(self, im_blob, img0):
        """
        Processes the image frame and finds bounding box(detections).

        Associates the detection with corresponding tracklets
        and also handles lost, removed, refound and active tracklets.

        Args:
            im_blob (np.array): Tensor of image. By default, shape of this tensor is [1, 3, 608, 1088].
            img0 (np.array): Input image sequence. By default, shape is [608, 1080, 3].

        Returns:
            output_stracks (list of TrackS): Information regarding the online_tracklets for the received image tensor.
        """
        self.frame_id += 1
        activated_stracks = []  # For storing active tracks, for the current frame.
        refind_stracks = []  # Lost Tracks whose detections are obtained in the current frame.
        lost_stracks = []  # The tracks which are not obtained in the current frame but are not removed.
        removed_stracks = []
        unconfirmed = []
        tracked_stracks = []  # type: list[TrackS]

        # Step 1: Network forward, get detections & embeddings
        if self.opt.infer310:
            pred = im_blob
        else:
            _, pred = self.model.predict(im_blob)
            pred = pred.asnumpy()
        # print('pred:',pred.sum())
        # Pred is tensor of all the proposals (default number of proposals: 54264).
        # Proposals have information associated with the bounding box and embeddings.
        pred = pred[pred[:, :, 4] > self.opt.conf_thres]
        # Pred now has lesser number of proposals. Proposals rejected on basis of object confidence score.

        if pred.size > 0:
            dets = non_max_suppression(np.expand_dims(pred, 0), self.opt.conf_thres, self.opt.nms_thres)[0]

            # Final proposals are obtained in dets. Information of bounding box and embeddings also included.
            # Next step changes the detection scales
            scale_coords(self.opt.img_size, dets[:, :4], img0.shape).round()

            # Detections is list of (x1, y1, x2, y2, object_conf, class_score, class_pred)
            # Class_pred is the embeddings.
            detections = [TrackS(TrackS.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                          (tlbrs, f) in zip(dets[:, :5], dets[:, 6:])]
        else:
            detections = []

        # Add newly detected tracklets to tracked_stracks
        for track in self.tracked_stracks:
            if not track.is_activated:
                # previous tracks which are not active in the current frame are added in unconfirmed list
                unconfirmed.append(track)
            else:
                # Active tracks are added to the local list 'tracked_stracks'
                tracked_stracks.append(track)

        self.tracking(
            activated_stracks,
            refind_stracks,
            lost_stracks,
            removed_stracks,
            unconfirmed,
            tracked_stracks,
            detections,
        )

        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


def joint_stracks(tlista, tlistb):
    """
    Append stracks.
    """
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    """
    Delete stracks.
    """
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    """
    Removes duplicate from stracks.
    """
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = [], []
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
