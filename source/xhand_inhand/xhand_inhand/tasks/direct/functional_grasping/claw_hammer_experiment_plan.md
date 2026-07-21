# Claw Hammer Experiment Plan: Pull Nail 3 cm, Then Hammer It Back

## 1. Core Research Idea

Use a **natural claw hammer** as the first experimental tool. The hammer has two functional regions:

- **Claw region**: pull / pry affordance, used to pull a nail upward.
- **Hammer face**: push / impact affordance, used to press or hammer the nail back in.
- **Handle region**: grasping region, used by the dexterous hand.

The key research question is:

> Can a dexterous hand switch between two natural affordances of the same tool through **in-hand reorientation**, under constrained arm motion?

The first task sequence is:

```text
Grasp claw hammer
→ expose claw affordance
→ engage nail head
→ pull nail out by 3 cm
→ rotate hammer in hand to expose hammer face
→ align hammer face with nail head
→ hammer / press nail back in
```

This is better than a ring-pulling task because the same target object, the nail, is used for both affordances: the claw pulls it out, and the hammer face pushes it back in.

---

## 2. Why This Task Is Suitable

This task naturally requires **affordance switching**:

- Pulling the nail requires the **claw** to be active.
- Driving the nail back requires the **hammer face** to be active.
- These two affordances are on opposite functional regions of the hammer.
- If the robot relies only on arm/wrist rotation, switching from claw to face may require a large motion.
- In a constrained-arm setup, the more reasonable solution is to rotate or roll the hammer inside the hand.

This makes the task a good first case for:

```text
functional grasping
+ in-hand tool reorientation
+ tokenized action representation
+ natural tool affordance switching
```

---

## 3. Experimental Setup

### Tool

Use a small real claw hammer.

Suggested properties:

- Short handle, suitable for the dexterous hand.
- Handle diameter large enough for stable grasping.
- Hammer head not too heavy.
- Claw opening large enough to engage a nail head.

If the real hammer is too difficult at first, wrap the handle with rubber tape to increase friction and diameter.

### Target Object

Use a nail-like object mounted in a controlled fixture.

Recommended first version:

- A large nail, pin, or custom nail-like peg.
- Nail head should be large enough for the claw to engage.
- Nail shaft should move vertically inside a guide hole.
- The fixture should provide adjustable friction or spring resistance.
- Required pull distance: **3 cm**.

Do not start with a real nail tightly embedded in wood. That may require excessive force and be unsafe for the hand. Start with a guided nail fixture that behaves like a nail but has controlled resistance.

### Fixture

A practical fixture can be:

```text
vertical guide block
+ movable nail/pin
+ spring or friction sleeve
+ scale/marker for measuring displacement
```

The fixture should allow:

- Initial nail height: fully inserted.
- Pulled state: nail lifted by 3 cm.
- Pressed state: nail pushed back near the original height.

---

## 4. Functional Geometry Graph

Represent the hammer as a graph:

```text
Tool graph = grasping region + functional regions + switching edges
```

### Nodes

| Node | Type | Geometry | Affordance |
|---|---|---|---|
| Handle | Grasping region | Cylinder / oval cylinder | Stable grasp, in-hand rotation |
| Claw | Functional region | Curved hook | Pull / pry |
| Hammer face | Functional region | Flat face | Push / hammer |

### Edges

| Edge | Meaning |
|---|---|
| Handle → Claw | Claw pose relative to grasping region |
| Handle → Hammer face | Face pose relative to grasping region |
| Claw → Hammer face | Required in-hand affordance switch |

The most important edge is:

```text
Claw affordance → Hammer-face affordance:
requires tool-in-hand rotation / reorientation
```

---

## 5. Task Phases

### Phase 1: Grasp Hammer

Initial condition:

- Hammer is already in the hand for the first MVP, or placed in a simple holder.
- The hand grasps the handle.

Success condition:

- Hammer remains in hand for a fixed duration.
- Tool slip is below threshold.

### Phase 2: Expose and Align Claw

Goal:

- Rotate/reorient the hammer so the claw is aligned with the nail head.

Success condition:

- Claw opening direction aligned with nail head.
- Claw tip near nail head.
- Arm motion remains within the constrained range.

### Phase 3: Engage Nail and Pull Out 3 cm

Goal:

- Use the claw to engage the nail head.
- Pull the nail upward by **3 cm**.

Success condition:

```text
nail_displacement >= 0.03 m
AND claw is engaged with nail head
AND hammer remains in hand
```

### Phase 4: In-Hand Switch from Claw to Hammer Face

Goal:

- Rotate/reorient the hammer in hand to expose the hammer face.

Success condition:

- Hammer face normal aligned with nail axis.
- Face center near nail head.
- Arm/wrist motion remains bounded.

### Phase 5: Hammer / Press Nail Back In

Goal:

- Use hammer face to push or hammer the nail back toward its initial depth.

First MVP can use quasi-static pressing rather than dynamic impact.

Success condition:

```text
nail_displacement <= 0.005–0.01 m from original inserted state
AND hammer face contacts nail head
AND non-functional tool regions do not drive the nail
```

---

## 6. Constrained-Arm Assumption

Do not fully lock the arm. Instead, use a constrained-arm setting:

```text
Arm translation: small residual motion only
Wrist rotation: limited range
Main affordance switching: tool-in-hand reorientation
```

Suggested limits:

- End-effector translation: ≤ 2 cm around nominal pose.
- Wrist rotation: ≤ 10–20 degrees.

This prevents the robot from solving the task by large arm flipping.

---

## 7. Action Representation

The task naturally supports tokenized actions.

Candidate tokens:

```text
StableGrasp
ExposeClaw
EngageNail
PullNail
LoosenGrip
RotateToolInHand
TightenGrip
ExposeHammerFace
AlignFace
PressNail
RecoverSlip
```

A practical controller can be hierarchical:

```text
High-level policy: selects token
Low-level controller: generates hand target trajectory + small arm residual
```

The token representation is meaningful because each token corresponds to a physical subskill in the hammer task.

---

## 8. Reward Sketch

Use phase-wise rewards.

### Pull Phase

```text
r_pull = nail_upward_progress
       + claw_engagement_reward
       + claw_alignment_reward
       - tool_slip_penalty
       - wrong_contact_penalty
```

### Switching Phase

```text
r_switch = - angle_error(hammer_face_frame, desired_face_frame)
         - tool_drop_penalty
         - excessive_arm_motion_penalty
```

### Press/Hammer Phase

```text
r_press = nail_downward_progress
        + face_contact_reward
        + face_alignment_reward
        - wrong_contact_penalty
        - tool_slip_penalty
```

Important: success should require the correct functional region to be active.

For example, nail-down progress should count only if the hammer face, not the claw or hand, contacts the nail head.

---

## 9. Metrics

Report phase-level and full-task metrics.

| Metric | Meaning |
|---|---|
| Grasp success | Hammer remains in hand |
| Claw alignment error | Claw frame vs nail head frame |
| Engagement success | Claw catches nail head |
| Pull distance | Nail displacement upward |
| Pull success | Nail pulled by 3 cm |
| In-hand switch success | Hammer face exposed after pull |
| Face alignment error | Hammer face normal vs nail axis |
| Press-back success | Nail returned near original depth |
| Tool slip | Hammer motion relative to palm |
| Arm motion magnitude | Verifies constrained-arm setting |
| Full sequence success | Pull 3 cm and press back in |

---

## 10. Baselines

Use baselines that test the necessity of each component.

1. **Fixed grasp, no in-hand switching**
   - Tests whether the task truly requires tool reorientation.

2. **Arm-only reorientation under constrained arm**
   - Tests whether the arm can solve affordance switching alone.

3. **Raw joint action RL**
   - Tests whether tokenized action helps.

4. **Token policy without functional graph**
   - Tests whether explicit affordance structure matters.

5. **Full method**
   - Functional graph + tokenized action + domain randomization.

---

## 11. Domain Randomization

Randomize functional geometry rather than exact tool identity.

### Hammer-side randomization

```text
handle radius / length
head mass
claw opening width
claw curvature
hammer face size
claw-face relative angle
friction
inertia
```

### Nail-fixture randomization

```text
nail head size
nail shaft friction
required pull force
spring stiffness
initial nail depth
nail orientation perturbation
```

The real-world test should use an actual small claw hammer and a physical nail fixture.

---

## 12. MVP Roadmap

### MVP-1: Tool Already in Hand

```text
hammer already grasped
→ claw alignment
→ pull nail 3 cm
```

### MVP-2: Add Affordance Switch

```text
pull nail 3 cm
→ rotate hammer in hand
→ expose hammer face
```

### MVP-3: Complete Sequence

```text
pull nail 3 cm
→ in-hand switch
→ press nail back in
```

### MVP-4: Add Grasping

```text
pick hammer from holder
→ complete full sequence
```

---

## 13. Main Risk

The biggest risk is force requirement. A real nail in wood may be too hard for the dexterous hand.

Therefore, the first experimental object should be a **nail-like guided pin with controlled resistance**, not a tightly embedded real nail.

Once the policy works, gradually increase realism:

```text
guided pin
→ low-friction nail fixture
→ soft foam / soft wood
→ real nail in wood
```
