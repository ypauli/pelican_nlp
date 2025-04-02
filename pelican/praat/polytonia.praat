# polytonia.praat -- include file for prosogram

# This file is included by prosogram.praat. It isn't a stand-alone script. Use prosogram.praat instead.
# Author: Piet Mertens
# For documentation see:
# http://sites.google.com/site/prosogram/
# http://sites.google.com/site/prosogram/polytonia
# http://sites.google.com/site/prosogram/userguide#polytonia

# 2021-12-10	bugfix
# Last modifications: 2020-02-27, 2021-12-10


# Procedure hierarchy
#
# polytonia_main
#       load_speaker_range		fetch pitch range parameters of current speaker into current context
#       get_tier_label_for_nucleus	get label for interval on specified tier for specified interval on nucleus tier 
#       get_values_for_nucleus
#       find_previous_nucleus	get index of preceding valid nucleus by same speaker
#    intrasyllabic_contour		determine contour type (level, rise, fall...) of current nucleus
#    polytonia_sustain			decide whether contour is labeled "sustain" 
#    polytonia_extrema			assign pitch level T (top) or B (bottom), when pitch is close to pitch range extrema
#    polytonia_localchange		assign pitch level L (low), M (mid), H (high), from pitch changes in nucleus and left context
#    polytonia_infer			assign pitch level from intrasyllabic contour and pitch range
#    polytonia_island 			assign pitch level in short interpausal stretch with unassigned pitch level, from pitch range
#       infer_pitchlevel		infer pitch level from pitch range
#    polytonia_plateau
# Debugging info added to contour label (partly discarded at end of process)
#    o							pitch out of valid range; don't use this syllable
#    l{							intrasyllabic large rise/fall starts in lower half of pitch range
#    m{							intrasyllabic large rise starts just below median of pitch range
#    h{                         intrasyllabic large fall starts in upper quarter of pitch range
#    l{{						intrasyllabic small rise starts slightly above level of previous nucleus, which is L
#    h4							intrasyllabic level contour in upper quarter of pitch range


; @logging_start: "reset debug", "_log.txt"


# Polytonia attemps to compute pitch level and pitch contour label for each syllable, using the prosodic properties (stored in table nucldatID) of these syllables.
# These labels are stored in a tier with syllable-sized or nucleus-sized intervals (depending on the availability of syllable boundaries). This tier is created in prosomain.praat, see "polytonia_annotation".
# The input intervals are provided by the Prosogram segmentation selected by the user (possibly based on corpus annotation). Pauses are marked as intervals with label "_".


procedure polytonia_main: .start_time, .end_time
; Compute Polytonia labels for speech in range <.start_time> .. <.end_time>.
   @debug_msg: "polytonia_main: entry"

  ; Initialize polytonia_tier
	  ; Use syllable-sized or nucleus-sized intervals for polytonia tiers
	  if (segm_type == segm_msyllvow or segm_type == segm_msyllpeak or segm_type == segm_mrhyme)	
		 .src_tier = syllable_tier	; use syllable boundaries for pitch labels
	  elsif (segm_type == segm_vnucl and segfile_available and syllables_available)
		 .src_tier = syllable_tier
	  else
		 .src_tier = nucleus_tier		; use nucleus boundaries for pitch labels
	  endif
	  selectObject: nucleiID
	  Remove tier: polytonia_tier		; This tier is empty when TextGrid is created
	  ; Copy intervals from <.src_tier> to polytonia_tier
	  Duplicate tier: .src_tier, polytonia_tier, "polytonia"
	  if (.src_tier == nucleus_tier)
		 @tier_merge_intervals_except: nucleiID, polytonia_tier, "a"
	  endif
	  @tier_clear_text: nucleiID, polytonia_tier

   current_speaker = 0		; no speaker identified yet; speakers are numbered from 1 to N; force initialization
   .ctxt_maxdur = 0.5		; max duration of left context used for detection of pitch changes
   ; Steps are applied to the nucleus intervals stored in nucleus_tier of nucleiID
   @intervals_from_time_range: nucleiID, nucleus_tier, .start_time, .end_time, "i_first", "i_last"
   @intrasyllabic_contour:	i_first, i_last, polytonia_tier
   @polytonia_sustain:		i_first, i_last, polytonia_tier
   @polytonia_extrema:		i_first, i_last, polytonia_tier
   @polytonia_localchange:	i_first, i_last, polytonia_tier, .ctxt_maxdur
   @polytonia_infer:		i_first, i_last, polytonia_tier, .ctxt_maxdur
   ; Duplicate tier for Polytonia version 2 
	  Remove tier: polytonia2_tier
	  Duplicate tier: polytonia_tier, polytonia2_tier, "polytonia-iw"
   @polytonia_extrapolate:	i_first, i_last, polytonia_tier, 1, .ctxt_maxdur
   @polytonia_extrapolate:	i_first, i_last, polytonia_tier, 0, .ctxt_maxdur
   @polytonia_plateau:		i_first, i_last, polytonia_tier
   @polytonia_extrapolate:	i_first, i_last, polytonia_tier, 0, .ctxt_maxdur
   @polytonia_postproc:		polytonia_tier, .start_time, .end_time

   ; Polytonia-iw (isolated words) adds step for short interpausal stretches with unassigned pitch level
   @intervals_from_time_range: nucleiID, nucleus_tier, .start_time, .end_time, "i_first", "i_last"
   @polytonia_island:		i_first, i_last, polytonia2_tier
   @polytonia_extrapolate:	i_first, i_last, polytonia2_tier, 1, .ctxt_maxdur
   @polytonia_extrapolate:	i_first, i_last, polytonia2_tier, 0, .ctxt_maxdur
   @polytonia_plateau:		i_first, i_last, polytonia2_tier
   @polytonia_extrapolate:	i_first, i_last, polytonia2_tier, 0, .ctxt_maxdur
   @polytonia_postproc:		polytonia2_tier, .start_time, .end_time
   @debug_msg: "polytonia_main: exit"
endproc


procedure load_speaker_range: .speaker
; Set global variables for use by following procedure calls.
; Variables in Hz: median, bottom, top, q1_, q3_ 
; Variables in ST: range_, large_, medium_, flat_, upper_range_, lower_range_
; Variables (other): current_speaker, nnucl_speaker 
   @debug_msg: "load_speaker_range: .speaker='.speaker'"
   if (.speaker <> current_speaker)
      current_speaker = .speaker
      selectObject: profileID
      .row = .speaker
      median = Get value: .row, j_pitch_median_Hz
      bottom = Get value: .row, j_pitch_bottom_Hz
      top = Get value: .row, j_pitch_top_Hz
      range_ = Get value: .row, j_pitch_range
      nnucl_speaker = Get value: .row, j_nrofnucl
      q1_ = Get value: .row, j_rawf0_p25
      q3_ = Get value: .row, j_rawf0_p75
      upper_range_ = 12 * log2 (top/median)
      lower_range_ = 12 * log2 (median/bottom)
      flat_ = 1.2			; ignored intersyllabic pitch interval
      if (range_ >= 8.5)	; "normal" range
         large_ = 4.5		; large intersyllabic pitch interval 
         medium_ = 3		; medium intersyllabic pitch interval
      elsif (range_ >= 7)	; narrow range
         large_ = 3.5		; large intersyllabic pitch interval 
         medium_ = 2.5		; medium intersyllabic pitch interval
      else					; very narrow range
         large_ = 3.2		; large intersyllabic pitch interval 
         medium_ = 2.5		; medium intersyllabic pitch interval
      endif
   endif
   @debug_msg: "load_speaker_range: exit"
endproc


procedure get_tier_label_for_nucleus: .i_nucl, .tier
; Return label of interval on tier <.tier> for nucleus interval with index <.i_nucl>.
; <result1> out: index into <.tier>, or 0 if not found
; <label$>	out: countour label
   selectObject: nucleiID
   .x1 = Get start time of interval: nucleus_tier, .i_nucl
   .x2 = Get end time of interval: nucleus_tier, .i_nucl
   result1 = Get interval at time: .tier, (.x1+(.x2-.x1)/2)
   label$ = Get label of interval: .tier, result1
endproc


procedure get_values_for_nucleus: .i
; Returns following values in global variables <pj>, <vf0lo>, <vf0hi>, <vf0start>, <vf0end>, <vhesit>
   selectObject: nucleiID
   .s$ = Get label of interval: pointer_tier, .i
   pj = number(.s$)
   selectObject: nucldatID
   ; lopitch, hipitch = f0 in Hz, after stylization
   vf0lo = Get value: pj, lopitch
   vf0hi = Get value: pj, hipitch
   vf0start = Get value: pj, j_f0_start
   vf0end = Get value: pj, j_f0_end
   vhesit = Get value: pj, j_hesitation
   vspeakerid = Get value: pj, j_speaker_id
endproc


procedure intrasyllabic_contour: .i_first, .i_last, .dest_tier
; Determine contour type (level, rise, fall...) of current nucleus
; <.i_first>...<.i_last>	indices within nucleus_tier, on which to apply prodecure
; <.dest_tier>				tier where output labels are stored
   @debug_msg: "intrasyllabic_contour: entry"
   for .j from .i_first to .i_last
      selectObject: nucleiID
	  @is_nucleus: .j
      if (result)
         .x1 = Get start time of interval: nucleus_tier, .j
         .x2 = Get end time of interval: nucleus_tier, .j
         @get_values_for_nucleus: .j
         @load_speaker_range: vspeakerid
         selectObject: stylID
         .i = Get nearest index from time: .x1
         .i2 = Get nearest index from time: .x2
         .label$ = ""
         .nsegments = .i2-.i		; nrof tonal segments
         while (.i < .i2)			; for each tonal segment
            selectObject: stylID
            .xL = Get time from index: .i
            .xR = Get time from index: .i+1
            .yL = Get value at time: .xL
            .yR = Get value at time: .xR
            .intST = 12 * log2 (.yR/.yL)	; pitch interval (in ST) in current tonal segment
            if (.intST >= large_)
               .label$ += "R"
            elsif (.intST >= medium_)
               .label$ += "r"
            elsif (.intST <= -large_)
               .label$ += "F"
            elsif (.intST <= -medium_)
               .label$ += "f"
            else
               .s$ = "_"
               if (.nsegments > 1 and .xR-.xL < 0.07)	; skip first short plateau !!
                  .s$ = ""
               endif
               .label$ += .s$	; level               
            endif
            .i += 1
         endwhile
         .label$ = replace_regex$(.label$, "[Rr][Rr]", "R", 0)         
         .label$ = replace_regex$(.label$, "[Ff][Ff]", "F", 0)         
         .label$ = replace_regex$(.label$, "___*", "_", 0)         
         ;.label$ = replace_regex$(.label$, "C.*", "C", 0)         
         selectObject: nucleiID
         .imid = Get interval at time: .dest_tier, .x1+(.x2-.x1)/2
         Set interval text: .dest_tier, .imid, .label$
      endif
   endfor
   @debug_msg: "intrasyllabic_contour: exit"
endproc


procedure polytonia_sustain: .i_first, .i_last, .dest_tier
; Decide whether syllabic contour receives label "S" (for "Sustain")
; <.dest_tier>    tier to be used for contour labels
   .mindur_sustain = 0.25
   .max_up = 1.5
   .max_down = -1.5
   for .j from .i_first to .i_last
      selectObject: nucleiID
      @is_nucleus: .j
      if (result)
         @get_values_for_nucleus: .j
         .dur = Get value: pj, j_nucldur
         .downST = Get value: pj, j_intrasyldown
         .upST = Get value: pj, j_intrasylup
         @get_tier_label_for_nucleus: .j, .dest_tier
         .contour_index = result1
         if (label$ = "_" and .dur >= .mindur_sustain and .downST > .max_down and .upST < .max_up)
; @debug_msg: "polytonia_sustain: dur='.dur:2' downST='.downST:1' upST='.upST:1'"
            selectObject: nucleiID
            Set interval text: .dest_tier, .contour_index, "S"
         endif
      endif
   endfor
endproc


procedure polytonia_localchange: .i_first, .i_last, .dest_tier, .ctxt_maxdur
; Assign pitch levels L (low), M (mid), or H (high) from pitch changes in nucleus and in left context.
; Obtain pitch extrema in nuclei in left context, where left context is
; - limited to 0.5 s (<.ctxt_maxdur>) : time between end of nucleus in left context and start of target nucleus
; - limited to 3 syllables (<.ctxt_width>)
; - restricted to syllables pronounced by same speaker
; Then assign H, M, or L depending on distance between start pitch and lower pitch in left context

   @debug_msg: "polytonia_localchange: entry"
   .ctxt_width = 3			; max nr of syllables in left context

   for .i from .i_first to .i_last
      selectObject: nucleiID
      .x1 = Get start time of interval: nucleus_tier, .i
      .x2 = Get end time of interval: nucleus_tier, .i
      @is_nucleus: .i
      if (result)
         speaker_interval = Get interval at time: speaker_tier, .x1
         @get_values_for_nucleus: .i
; @msg: "-- t='.x1:3' pj='pj', 'vf0lo:1', 'vf0hi:1', 'vf0start:1', hes='vhesit', sp='vspeakerid'"
         .f0_start_i = vf0start		; will be used when left context does not contain valid nuclei
         @load_speaker_range: vspeakerid
         .ctxt_lo = 1000	; initialize lower F0 value in context (to a very high value)
         .ctxt_hi = 0		; initialize higher F0 value in context (to a very low value)
         .ctxt_nsyll = 0	; nrof syllables in context
         .t2 = .x1			; initialize!! <.t2> will hold end of previous nucleus in repeat loop
         .j = .i
         repeat				; estimate pitch extrema in nuclei in left context
            .j -= 1			; preceeding interval in tier
            if (.j >= .i_first)
               selectObject: nucleiID
               @is_nucleus: .j
               if (result)
                  .t1 = Get start time of interval: nucleus_tier, .j    ; starttime of preceeding nucleus
                  .t2 = Get end time of interval: nucleus_tier, .j      ; endtime of preceeding nucleus
                  speaker_interval_j = Get interval at time: speaker_tier, .t1
                  if (.x1 - .t2 < .ctxt_maxdur 
                     ... and speaker_interval == speaker_interval_j)	; same speaker
                     @get_values_for_nucleus: .j
                     if (not vhesit)									; discard data from hesitations
                        if (vf0lo > bottom)
                           .ctxt_lo = min (.ctxt_lo, vf0lo)
                        endif
                        if (vf0hi < top)
                           .ctxt_hi = max (.ctxt_hi, vf0hi)
                        endif
                        if (vf0lo > bottom and vf0hi < top)
                           .ctxt_nsyll += 1
                        endif
                     endif
                  endif
               endif
            endif
         until (.ctxt_nsyll == .ctxt_width or (.x1-.t2 >= .ctxt_maxdur) or .j <= .i_first)
         if (.ctxt_nsyll == 0)		; no left context found 
             .ctxt_lo = .f0_start_i
             .ctxt_hi = .f0_start_i
         endif
         @debug_msg: "polytonia_localchange: context found LO='.ctxt_lo:1' HI='.ctxt_hi:1' N='.ctxt_nsyll' <--> f0start='.f0_start_i:0'"
      
         selectObject: nucleiID
         .jct = Get interval at time: .dest_tier, .x1+(.x2-.x1)/2
         .label$ = Get label of interval: .dest_tier, .jct
         if (index_regex (.label$, "^[_RrFfS]"))	; Level or Rise or Fall or Sustain
            if (.ctxt_lo == undefined or .ctxt_lo < bottom or .ctxt_hi > top)
               ; don't assign pitch level
            else			; assign pitch level H or M, using pitch interval above pitch minimum in left context
               .distST = 12 * log2 (.f0_start_i/.ctxt_lo)
               if (.distST >= large_)
                  .label$ = "H" + .label$
               elsif (.distST >= medium_)
                  .label$ = "M" + .label$
               else 		; assign pitch level L, using pitch interval below pitch maximum in left context
                  if not (.ctxt_hi == undefined or .ctxt_hi < bottom)
                     .distST = 12 * log2 (.f0_start_i/.ctxt_hi)
                     if (.distST <= -large_)
                        .label$ = "L" + .label$
                     endif
                  endif
               endif
            endif
         endif
         Set interval text: .dest_tier, .jct, .label$
      endif
   endfor
   @debug_msg: "polytonia_localchange: exit"
endproc


procedure polytonia_extrema: .i_first, .i_last, .tier
; Assign pitch levels Bottom or Top to contour label, when tonal segment starts or ends in these extreme levels. 
; Also adds creak symbol 
; <.tier>    tier to be used for contour info
   for .i from .i_first to .i_last
      @get_tier_label_for_nucleus: .i, .tier
      .contour_index = result1
      .seed$ = label$
      @is_nucleus: .i
      if (result)
         @get_values_for_nucleus: .i
         pv_hi = vf0hi
         pv_lo = Get value: pj, f0_min
         @load_speaker_range: vspeakerid
         selectObject: nucleiID
         if (12 * log2 (pv_lo/bottom) <= -4) or (12 * log2 (pv_hi/top) >= 8)	; out of range -> will block pitch level interpretation
            label$ = "o"
         else
            if (index_regex (.seed$, "[Ff]_?$") and vf0end <= bottom)			; fall ends at bottom
               label$ += ",B"
            endif
            if (index_regex (.seed$, "^[_Rr]") and vf0start <= bottom)			; starts at bottom
               label$ = "B" + label$
            endif
            if (range_ >= 11 and nnucl_speaker >= 200 and upper_range_ >= 8)
               if (12 * log2 (vf0start/top) >= 0)                      			; starts at top
                  label$ = "T" + label$
               elsif (index_regex (.seed$, "[Rr_F]$") and (12 * log2 (pv_hi/top) >= 2))		; rise ends at top
                  label$ += ",T"
               endif
            endif
            @get_creak_for_nucleus: .i
            if (result)
               label$ = "C" + label$
            endif
         endif
         Set interval text: .tier, .contour_index, label$
      endif
   endfor
endproc


procedure polytonia_infer: .i_first, .i_last, .dest_tier, .ctxt_maxdur
; Infer pitch level from intrasyllabic variation and speaker pitch range
; <.dest_tier>    in: tier to be used for contour label
   for .i from .i_first to .i_last
      selectObject: nucleiID
      @is_nucleus: .i
      if (result)
         @get_tier_label_for_nucleus: .i, .dest_tier
         .contour_index = result1
         .seed$ = label$
         if (index_regex (.seed$, "^[_r]*[RFrf]"))				; no pitch level assigned
            .unassigned = 1
            @get_values_for_nucleus: .i
            @load_speaker_range: vspeakerid
            selectObject: nucleiID
            if (index_regex (.seed$, "^[_r]*R"))				; pitch level unassigned AND large rise 
               if (vf0start <= median)							; rise starts in lower half of range
                  .label$ = "l{" + .seed$
                  .unassigned = 0
               elsif (vf0start <= (median + (top-median/4)))	; rise starts just above median
                  .label$ = "m{" + .seed$
                  .unassigned = 0
               endif
            endif
            if (.unassigned and index_regex (.seed$, "^[_f]*F")); pitch level unassigned AND large fall
               if (vf0end <= bottom and vf0start < median)		; towards bottom, in lower half of range
                  .label$ = "l{" + .seed$
                  .unassigned = 0
               elsif (vf0start >= (median + (top-median)/2))	; fall starts in upper quarter of range
                  .label$ = "h{" + .seed$
                  .unassigned = 0
               endif
            endif 
            if (.unassigned and index_regex (.seed$, "^r"))		; pitch level unassigned AND small rise
               @find_previous_nucleus: .i, .i_first, .ctxt_maxdur
               if (result)
                  @get_tier_label_for_nucleus: result, .dest_tier
                  if (index_regex (label$, "^L"))				; polytonia label of previous syllable
                     @get_values_for_nucleus: result
                     .distST = 12 * log2 (vf0start/result3)		; result3 = f0_start of prev syllable 
                     if (.distST >= 0 and .distST < medium_)	; slightly above prev syll which is L
                        selectObject: nucleiID
                        .label$ = "l{{" + .seed$
                        .unassigned = 0
                     endif
                  endif
               endif
            endif 
            if (.unassigned and index_regex (.seed$, "^_"))		; pitch level unassigned AND level
               if (vf0start >= (median + (top-median)/2))		; starts in upper quarter of range
                  .label$ = "h4" + .seed$
                  .unassigned = 0
               endif
            endif 
            if (.unassigned == 0)	; success
               Set interval text: .dest_tier, .contour_index, .label$
            endif 
         endif
      endif
   endfor
endproc


procedure infer_pitchlevel: .iNucl, .dest_tier
; Infer pitch level from pitch range. 
; <.iNucl>		index of syllable in nucleus tier of nucleiID
; <.dest_tier>	tier where pitch label is written  
      .unassigned = 0
      selectObject: nucleiID
      label$ = Get label of interval: nucleus_tier, .iNucl
      @is_nucleus: .iNucl
      if (result)
         @get_tier_label_for_nucleus: .iNucl, .dest_tier
         .contour_index = result1
         .seed$ = label$
         if (index_regex (.seed$, "^[_RFrfSC]"))	; no pitch level assigned
            .unassigned = 1
            .s$ = Get label of interval: pointer_tier, .iNucl
            .p = number (.s$)
            selectObject: nucldatID
            .v = Get value: .p, j_f0_mean
            speaker_j = Get value: .p, j_speaker_id 
            @load_speaker_range: speaker_j	
            ; median, top, bottom, q1_, q3_ in Hz;  range, upper_range_, lower_range_, large_, medium_, flat_ in ST
            selectObject: nucleiID
            if (.v >= q3_)
               label$ = "h" + .seed$
               .unassigned = 0
            elsif (.v < q3_ and .v > q1_)
               label$ = "m" + .seed$
               .unassigned = 0
            else
               label$ = "l" + .seed$
               .unassigned = 0
            endif
            if (not .unassigned)	; success
               Set interval text: .dest_tier, .contour_index, label$
            endif 
         endif
      endif
endproc


procedure polytonia_island: .i_first, .i_last, .dest_tier
; For a short (up to 5 syllables) interpausal stretch with unassigned pitch level, assign pitch level on the basis of speaker pitch range.
; Find syllable with maximum for mean pitch.
   @debug_msg: "polytonia_island: entry"
   .i = .i_first
   while (.i < .i_last)
      selectObject: nucleiID
      @is_nucleus: .i
      if (result)
         @get_values_for_nucleus: .i
         .after_pause = Get value: pj, j_after_pause
         .before_pause = Get value: pj, j_before_pause
         .mean_max = Get value: pj, j_f0_mean
         if (.after_pause)
            .first_nucl = .i		; first nucleus in interpausal stretch
            .nrofvalid = 0			; nr of syllabic nuclei in stretch
            .ok = 1
            .speaker_id_prev = vspeakerid
            .i -= 1
            repeat
               .i += 1
               @is_nucleus: .i
               .t1 = Get start time of interval: nucleus_tier, .i
               if (result)
                  .nrofvalid += 1
                  @get_values_for_nucleus: .i
                  .before_pause = Get value: pj, j_before_pause
                  .speaker_id = vspeakerid
                  .pv_f0mean = Get value: pj, j_f0_mean
                  if (.nrofvalid = 1 or .pv_f0mean > .mean_max)
                     .i_mean_max = .i		; index in nucleiID for nucleus in stretch with max pitch 
                     .mean_max = .pv_f0mean
                  endif
                  @get_tier_label_for_nucleus: .i, .dest_tier
                  .unassigned = index_regex (label$, "^[_RFrfSC]")
                  .last_nucl = .i			; currently last nucleus in stretch
                  .ok = (.speaker_id == .speaker_id_prev) and (.nrofvalid <= 5) and .unassigned
               endif
@debug_msg: "polytonia_island: i='.i' ('.t1:3') contour_label='.contour_label$' unassigned='.unassigned' ok='.ok'"
            until ((not .ok) or .before_pause)
            if (.ok and .before_pause)
               @infer_pitchlevel: .i_mean_max, .dest_tier
;               selectObject: nucleiID
;               .t1 = Get start time of interval: nucleus_tier, .first_nucl
;               .t2 = Get end time of interval: nucleus_tier, .last_nucl
; @debug_msg: "polytonia_island: *** Island FOUND from '.t1:3' to '.t2:3'"
            endif
	     endif ; if after pause
      endif ; if is_nucleus
      .i += 1
   endwhile
   @debug_msg: "polytonia_island: exit"
endproc


procedure polytonia_extrapolate: .i_first, .i_last, .dest_tier, .backward, .ctxt_maxdur
; Given a syllable with pitch level assigned, assign pitch level to neighbouring unassigned syllable
; <.dest_tier>    tier to be used for contour info
   if (.backward)
      .i = .i_last
      .stop_interval = .i_first
      .step = -1
   else
      .i = .i_first
      .stop_interval = .i_last
      .step = 1
   endif
   while (.backward and .i > .stop_interval) or (not .backward and .i < .stop_interval)
      @is_nucleus: .i
      if (result)
         @get_values_for_nucleus: .i
         if (not vhesit)
            @get_tier_label_for_nucleus: .i, .dest_tier
            .seed$ = label$
            if (index_regex (.seed$, "^[HhMmLlBb]"))				; pitch level assigned
               .t = Get start time of interval: nucleus_tier, .i
               .t2 = Get end time of interval: nucleus_tier, .i
               speaker_interval = Get interval at time: speaker_tier, .t
               .pv_start = vf0start
               @get_values_for_nucleus: .i
               @load_speaker_range: vspeakerid
               .success = 0			; 1 when managed to assign an unassigned syllable
               .found_assigned = 0	; while extrapolating, encountered syllable with pitch level assigned
               .k = 0				; distance (in nrof nuclei) between assigned and unassigned
               selectObject: nucleiID
               repeat				; find neighbouring nuclei with unassigned pitch level
                  .k += 1
                  .other_nucl = .i + .step * .k
                  if ((.backward and .other_nucl >= .stop_interval) or (not .backward and .other_nucl <= .stop_interval))
                     label$ = Get label of interval: nucleus_tier, .other_nucl
                     .t1o = Get start time of interval: nucleus_tier, .other_nucl
                     .t2o = Get end time of interval: nucleus_tier, .other_nucl
                     if (.backward)
                        .dt = .t - .t2o
                     else
                        .dt = .t1o - .t2
                     endif
                     @is_nucleus: .other_nucl
                     if (result)
                        speaker_interval_j = Get interval at time: speaker_tier, .t1o
                        if (.dt < .ctxt_maxdur
                           ... and speaker_interval == speaker_interval_j)	; same speaker
                           @get_tier_label_for_nucleus: .other_nucl, .dest_tier
                           .contour_index = result1
                           if (index_regex (label$, "^[_RrFfS]")) ; no pitch level assigned
							  ; next call stores label in label$
                              @get_values_for_nucleus: .other_nucl
                              .distST = 12 * log2 (.pv_start/vf0start)
                              .success = 1			; assume success
					 		  .lev$ = ""			; new pitch level
                              selectObject: nucleiID
                              if (index_regex (.seed$, "^[Hh]") and .distST >= large_)	; UP
                                 .lev$ = "l"
                              ;elsif (index_regex (.seed$, "^[Hh]") and .distST >= medium_)
                              ;   .lev$ = "m"
                              elsif (index_regex (.seed$, "^[Mm]") and .distST >= medium_)
                                 .lev$ = "l"
                              elsif (index_regex (.seed$, "^[Mm]") and .distST <= -medium_)		; DOWN
                                 .lev$ = "h"
                              elsif (index_regex (.seed$, "^[Ll]") and .distST <= -large_)
                                 .lev$ = "h"
                              elsif (index_regex (.seed$, "^[Ll]") and .distST <= -medium_)
                                 .lev$ = "m"
                              elsif (index_regex (.seed$, "^[Bb]") and .distST <= -medium_ 
                                 ... and .distST > -large_ and .backward)						; down to BOTTOM
                                 .lev$ = "l"
                              elsif (index_regex (.seed$, "^[Ll]") and abs(.distST) <= flat_)	; SAME LEVEL
                                 .lev$ = "l"
                              elsif (index_regex (.seed$, "^[Mm]") and abs(.distST) <= flat_)
                                 .lev$ = "m"
                              elsif (index_regex (.seed$, "^[Hh]") and abs(.distST) <= flat_)
                                 .lev$ = "h"
                              else
                                 .success = 0		; unable to extrapolate
                              endif
                              Set interval text: .dest_tier, .contour_index, .lev$ + label$
                           else
                              .found_assigned = 1
                           endif
                        endif
                     endif
                  endif
               until (.other_nucl = .stop_interval or .dt >= .ctxt_maxdur or .success or .found_assigned)
            endif
         endif ; no hesitation
      endif ; valid nucleus
      .i += .step
   endwhile
endproc


procedure polytonia_plateau: .i_first, .i_last, .dest_tier
; Given a series of syllables without pitch change, attribute pitch level
; <.dest_tier>    tier to be used for contour info
   .i = .i_first
   while (.i < .i_last)
      @is_nucleus: .i
      if (result)
         s$ = Get label of interval: pointer_tier, .i
         @get_values_for_nucleus: .i
         if (not vhesit)
            @get_tier_label_for_nucleus: .i, .dest_tier
            .contour_index = result1
            .seed$ = label$
            if (index_regex (.seed$, "^[_RrFfS]"))	; first syll of sequence, pitch level unassigned
               .nsyll = 1
               .t1 = Get start time of interval: nucleus_tier, .i
               .t2 = Get end time of interval: nucleus_tier, .i
               .speaker_interval = Get interval at time: speaker_tier, .t1
               @get_values_for_nucleus: .i
               .pv_start = vf0start
               @load_speaker_range: vspeakerid
               .success = 1				; 
               .end = 0					; encountered end condition
               .other_nucl = .i
               repeat					; check following nuclei which are unassigned
                  selectObject: nucleiID
                  .other_nucl += 1
                  if (.other_nucl <= .i_last)
                     .t1o = Get start time of interval: nucleus_tier, .other_nucl
                     .t2o = Get end time of interval: nucleus_tier, .other_nucl
                     @is_nucleus: .other_nucl
                     if (result)
                        .speaker_interval_j = Get interval at time: speaker_tier, .t1o
                        if (.t1o - .t2 <= 0.2
                           ... and .speaker_interval == .speaker_interval_j)	; same speaker
                           @get_tier_label_for_nucleus: .other_nucl, .dest_tier
                           if (index_regex (label$, "^_$")) ; no pitch level assigned AND level
                              @get_values_for_nucleus: .other_nucl
                              .distST = 12 * log2 (.pv_start/vf0start)
                              if (abs(.distST) > flat_)
                                 .success = 0
                              else
                                 .nsyll += 1
                              endif
                              if (.success and .nsyll >= 3)
                                 selectObject: nucleiID
                                 .distST = 12 * log2 (.pv_start/median)
                                 if (.pv_start < median and (12 * log2 (.pv_start/bottom) >= 1))
                                    Set interval text: .dest_tier, .contour_index, "L" + .seed$
                                    .end = 1
                                    .i = .other_nucl	; skip syllables of plateau
                                 elsif (.distST >= 1 and .distST <= upper_range_/2)
                                    Set interval text: .dest_tier, .contour_index, "M" + .seed$
                                    end = 1
                                    .i = .other_nucl	; skip syllables of plateau
                                 endif
                              endif
                           else	; pitch level assigned for this syllable
                              .end = 1
                              .i = .other_nucl	; skip syllables
                           endif
                           .t2 = .t2o
                        else
                           .end = 1
                           .i = .other_nucl	; skip syllables
                        endif
                     endif
                  endif ; .other_nucl <= .i_last
               until (.other_nucl = .i_last or (not .success) or .end)
            endif ; unassigned
         endif ; no hesitation
      endif
      .i += 1
   endwhile
endproc


procedure polytonia_postproc: .tier, .start_time, .end_time
; Post-processing of Polytonia tier
   @intervals_from_time_range: nucleiID, .tier, .start_time, .end_time, "j_first", "j_last"
   for .j from j_first to j_last
      s$ = Get label of interval: .tier, .j
      s$ = replace_regex$(s$, "[{4o]", "", 0)					; remove debugging comments
      s$ = replace_regex$(s$, "([lmhb])", "\U\1", 0)			; to uppercase
      s$ = replace_regex$(s$, "^(C?[LlMmHhBbT]?)_$", "\1", 1)	; remove single trailing "_"
      s$ = replace_regex$(s$, "FF", "F", 1)	                    ; simplify double fall
      Set interval text: .tier, .j, s$
   endfor
endproc


procedure find_previous_nucleus: .current_nucleus, .stop_interval, .ctxt_maxdur
; Find index of preceding nucleus of same speaker and within time range
; <.stop_interval>	lower index into nucleiID
; <result> index of previous nucleus or 0 if not found 
   .found = 0
   selectObject: nucleiID
   .t = Get start time of interval: nucleus_tier, .current_nucleus
   .speaker_interval = Get interval at time: speaker_tier, .t
   .prev = .current_nucleus
   repeat
      .prev -= 1
      .dt = .ctxt_maxdur
      if (.prev >= .stop_interval)
         selectObject: nucleiID
         .t1 = Get start time of interval: nucleus_tier, .prev
         .t2 = Get end time of interval: nucleus_tier, .prev
         .dt = .t-.t2
         @is_nucleus: .prev
         if (result)
            .speaker_interval_j = Get interval at time: speaker_tier, .t1
            if (.dt < .ctxt_maxdur and .speaker_interval == .speaker_interval_j)	; same speaker
               .found = 1
            endif
         endif
      endif
   until (.prev <= .stop_interval or .dt >= .ctxt_maxdur or .found)
   if (.found) 
      result = .prev
   else
      result = 0
   endif
endproc


procedure get_creak_for_nucleus: .i_nucl
; <result> return 1 if nucleus is creaky 
   result = 0
   selectObject: nucleiID
   .x = Get start time of interval: nucleus_tier, .i_nucl
   .x2 = Get end time of interval: nucleus_tier, .i_nucl
   .n = Get number of intervals: creak_tier
   .i = Get interval at time: creak_tier, .x
   repeat
      .s$ = Get label of interval: creak_tier, .i
      if (index_regex (.s$, "^[Cc]"))
         result = 1
      else
         .x = Get end time of interval: creak_tier, .i 
         .i += 1
         if (.i > .n)
            .x = .x2
         endif
      endif
   until (result or .x >= .x2)
endproc

