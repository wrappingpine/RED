# Tasks: Better Cursor Coordination

**Input**: Design documents from `/specs/001-better-cursor-coordination/`

**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Test tasks are included as requested by the development workflow (Unit/Integration/Property-based/Synthetic testing).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Verify specs directory structure matches plan.md (artifacts already created by Spec Kit scaffolding)
- [X] T002 Configure structured JSON logging for pipeline stage timings in `airmouse/debug/diagnostics.py` with fields: `stage_name`, `duration_ms`, `timestamp`, `frame_id` (FR-009)
- [X] T003 [P] Verify development virtual environment is active and all current tests pass: `pytest airmouse/tests/`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure and configuration schema updates that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T004 Add head-relative and projection parameters to `airmouse/config/schema.py` including `use_head_relative`, `virtual_plane_distance`, `virtual_plane_width`, `virtual_plane_height`, `head_confidence_threshold`, `projection_smoothing`, `one_euro_min_cutoff`, `one_euro_beta`, `one_euro_d_cutoff`, and `reference_point_update_mode`
- [X] T005 [P] Update profile defaults in `airmouse/config/profiles.py` with schema-validated values
- [X] T006 Implement coordinate space contracts and validation utility in `airmouse/control/coordinate_contract.py`
- [X] T006a [P] Create `SmoothingFilter` abstract base class in `airmouse/control/smoothing.py` with `OneEuroFilter` and `EmaFilter` concrete implementations, both operating on normalized plane (u,v) coordinates (FR-002)

**Checkpoint**: Configuration schema and coordinate contracts updated. Proceed to user story implementation.

---

## Phase 3: User Story 1 - Natural Hand-to-Cursor Mapping (Priority: P1) 🎯 MVP

**Goal**: Establish directional correctness and basic hand-to-cursor mapping without mirroring or axis inversion.

**Independent Test**: Run synthetic landmark tests and verify hand-to-cursor mapping directions match (SC-001).

### Tests for User Story 1
- [X] T007 [P] [US1] Create unit tests in `airmouse/tests/test_projection.py` to verify synthetic hand movements in all 4 cardinal directions and diagonals match expected cursor directions
- [X] T008 [P] [US1] Create unit tests in `airmouse/tests/test_projection.py` to verify axis inversion (`invert_x`, `invert_y`) behavior

### Implementation for User Story 1
- [X] T009 [US1] Update `airmouse/control/cursor.py` to parse axis inversion config and correctly map coordinates
- [X] T010 [US1] Implement coordinate translation logic in `airmouse/control/coordinate_contract.py` to map from [0,1] plane coordinates to screen pixels
- [X] T011 [US1] Ensure cursor coordinates are clamped to screen boundaries in `airmouse/control/cursor.py` (FR-008)
- [X] T011a [US1] Register global hotkey Super+Alt+M in `airmouse/control/hotkeys.py` for runtime toggle between head-relative and legacy mapping modes (FR-006, Principle X)

**Checkpoint**: Natural hand-to-cursor mapping is fully functional and testable independently.

---

## Phase 4: User Story 2 - Head-Relative Virtual Plane Projection (Priority: P1)

**Goal**: Implement projection of index fingertip through eye midpoint onto the virtual plane before smoothing (FR-001).

**Independent Test**: Verify head-movement invariance (SC-002): cursor drift <2px when head moves but hand remains fixed relative to head.

### Tests for User Story 2
- [X] T012 [P] [US2] Implement unit test `test_head_movement_invariance` in `airmouse/tests/test_projection.py` using synthetic face and hand poses
- [X] T013 [P] [US2] Implement fallback unit test in `airmouse/tests/test_projection.py` verifying graceful degradation to legacy camera coordinates when face tracking is lost (FR-010)

### Implementation for User Story 2
- [X] T014 [US2] Update `airmouse/vision/virtual_plane.py` to calculate ray intersection with the virtual plane using head coords from face tracker
- [X] T015 [US2] Modify `airmouse/vision/tracking_processor.py` to execute projection *before* applying any smoothing filters (Bug 1 fix, FR-001)
- [X] T016 [US2] Implement auto-pause / auto-fallback to camera mapping in `airmouse/vision/tracking_processor.py` when face confidence falls below threshold (FR-010, SC-006)

**Checkpoint**: Head-relative virtual plane projection works correctly with head-movement invariance.

---

## Phase 5: User Story 3 - Smooth Cursor with One Euro Filter on Plane Coordinates (Priority: P1)

**Goal**: Apply smoothing (One Euro Filter) directly to plane coordinates (u,v) instead of raw landmarks.

**Independent Test**: Verify jitter reduction (SC-004) under synthetic Gaussian noise (std dev <1px).

### Tests for User Story 3
- [ ] T017 [P] [US3] Implement unit test `test_plane_coordinate_smoothing` in `airmouse/tests/test_cursor_smoothing.py` to verify One Euro Filter behaves adaptively on plane coordinates
- [ ] T018 [P] [US3] Implement unit test `test_jitter_reduction` in `airmouse/tests/test_cursor_smoothing.py` feeding synthetic noisy inputs

### Implementation for User Story 3
- [ ] T019 [US3] Modify `airmouse/vision/tracking_processor.py` to apply the One Euro Filter to normalized plane (u,v) coordinates instead of raw landmarks
- [ ] T020 [US3] Update `airmouse/control/cursor.py` to execute EMA smoothing on plane coordinates when One Euro Filter is disabled
- [ ] T021 [US3] Log pipeline stage timings in `airmouse/debug/diagnostics.py` to measure latency (FR-009)

**Checkpoint**: Smoothing is performed at the correct pipeline stage on plane coordinates.

---

## Phase 6: User Story 4 - Correct Reference Point Update for Head-Relative Mode (Priority: P2)

**Goal**: Fix Bug 2 by updating the relative movement reference point every frame in head-relative mode.

**Independent Test**: Verify relative cursor tracking works inside the dead zone in head-relative mode.

### Tests for User Story 4
- [ ] T022 [P] [US4] Implement unit test `test_reference_point_every_frame` in `airmouse/tests/test_head_relative.py` for head-relative reference point updates
- [ ] T023 [P] [US4] Implement unit test `test_reference_point_dead_zone_exit` in `airmouse/tests/test_head_relative.py` for legacy dead zone updates

### Implementation for User Story 4
- [ ] T024 [US4] Modify reference point update logic in `airmouse/vision/tracking_processor.py` to update every frame when `use_head_relative` is True (Bug 2 fix, FR-003)
- [ ] T025 [US4] Preserve legacy reference point update logic in `airmouse/vision/tracking_processor.py` when `use_head_relative` is False (FR-004)

**Checkpoint**: Correct reference point behavior for both tracking modes.

---

## Phase 7: User Story 5 - Velocity Limiter Fix (Priority: P2)

**Goal**: Fix Bug 3 by providing a velocity limiter that correctly tracks positions over time instead of treating input deltas as positions (FR-005).

**Independent Test**: Verify velocity is capped precisely at configured `max_velocity` limits (SC-005).

### Tests for User Story 5
- [ ] T026 [P] [US5] Implement unit test `test_velocity_capping` in `airmouse/tests/test_velocity_limiter.py` with synthetic high-speed inputs
- [ ] T027 [P] [US5] Implement unit test `test_no_overshoot` in `airmouse/tests/test_velocity_limiter.py` verifying stable boundary capping

### Implementation for User Story 5
- [ ] T028 [US5] Create a new `VelocityLimiter` class in `airmouse/control/velocity_limiter.py` that computes velocity `v = (pos - prev_pos) / dt` and limits magnitude (Bug 3 fix)
- [ ] T029 [US5] Integrate the new `VelocityLimiter` into the main tracking processor loop in `airmouse/vision/tracking_processor.py` replacing old limiter logic

**Checkpoint**: Velocity limiter data contract fixed and successfully integrated.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Performance optimization, validation, and documentation updates across all features.

- [ ] T030 Documentation updates in `specs/001-better-cursor-coordination/quickstart.md`
- [ ] T031 Run entire validation suite and verify all unit, integration, and property tests pass (SC-007)
- [ ] T032 Verify latency performance budget is met (<50ms at 30 FPS) via `benchmark_full.py` (SC-003)
- [ ] T033 Code cleanup, refactoring, and removal of any temporary test debug files from git tracker
- [ ] T034 [P] Add CI latency gate: fail benchmark if P95 end-to-end latency > 50ms at 30 FPS (SC-003, Principle IV)
- [ ] T035 [P] Verify no `print()` statements in production code paths: `grep -r 'print(' airmouse/ --include='*.py' | grep -v test` (Principle XVII)
- [ ] T036 [US2] Handle virtual plane intersection failure edge case in `airmouse/vision/virtual_plane.py`: clamp to nearest plane boundary, log structured WARNING with tag `projection_intersection_failed`, continue tracking without interruption

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-7)**: All depend on Foundational phase completion
  - Phase 3 (US1) -> Phase 4 (US2) -> Phase 5 (US3) -> Phase 6 (US4) -> Phase 7 (US5)
- **Polish (Final Phase)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Natural direction mapping. Pre-requisite for all other stories.
- **User Story 2 (P2)**: Virtual plane projection. Depends on US1 coordinates.
- **User Story 3 (P3)**: Plane coordinate smoothing. Depends on US2 projection.
- **User Story 4 (P4)**: Reference point fix. Depends on US2 and US3.
- **User Story 5 (P5)**: Velocity limiter fix. Can run in parallel with US4 once US1-3 are complete.

### Parallel Opportunities

- Phase 1 Setup tasks T001-T003 can run in parallel.
- Test suite setup (T007-T008, T012-T013, T017-T018, T022-T023, T026-T027) can be authored in parallel by test-focused agents.
- Models and math calculations (T009, T014, T028) can be developed independently.

---

## Parallel Example: User Story 2

```bash
# Author tests for virtual plane projection and fallback concurrently:
Task: "Implement test_head_movement_invariance in airmouse/tests/test_projection.py"
Task: "Implement fallback unit test in airmouse/tests/test_projection.py verifying graceful degradation"
```

---

## Implementation Strategy

### MVP First (User Story 1-3)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3 (US1), Phase 4 (US2), and Phase 5 (US3)
4. **STOP and VALIDATE**: Run `quickstart.md` Scenario 1, 2, 3, 4. This forms the primary MVP!

### Incremental Delivery

1. Setup + Foundational ready
2. Deliver US1 (natural mapping) → Validate basic cursor
3. Deliver US2 (projection) + US3 (smoothing) → Validate head invariance and jitter → MVP!
4. Deliver US4 (reference point) → Validate relative movement within dead zone
5. Deliver US5 (velocity limiter) → Validate velocity capping
6. Deliver final polish and benchmarks → Release ready!