# SGAS-v2 Explained in Very Simple Terms

## The main idea

Imagine that the left leg and the right leg are two drummers.

When a person walks evenly, the two drummers usually play with similar timing, strength, steadiness, and rhythm. After a stroke, the two drummers may become less similar. One may move more slowly, more weakly, less consistently, or with a different movement shape.

The program listens to these two “drummers,” measures how different they are, and gives the person one overall asymmetry score.

This score is about what the sensors see. It is not a doctor's clinical stroke score.

## Step 1: Find every participant

The program looks inside two main folders:

- Healthy
- Stroke

Inside each participant folder, it expects four recordings:

- left-leg accelerometer;
- left-leg gyroscope;
- right-leg accelerometer;
- right-leg gyroscope.

If an important file is missing, the program cannot process that participant normally.

## Step 2: Make the recordings start together

The four sensors may not begin at exactly the same moment.

Imagine that four people recorded the same race, but one pressed the record button a little earlier and another pressed it a little later. Before comparing the videos, we first cut them so they all begin and end at the same moments.

The program does the same thing with the sensors. It creates one shared clock with 100 measurements every second. It does not invent information across large empty gaps.

This is important because a timing mistake could look like a difference between the legs even when it is only a recording problem.

## Step 3: Clean away small sensor noise

Raw sensor signals are a little shaky, like a video recorded with a trembling hand.

The program gently smooths the left and right gyroscope signals. It only smooths good parts of the recording and does not connect large missing sections.

## Step 4: Find the walking cycles

A walking cycle is like one repeated beat in the walking rhythm.

The program finds strong gyroscope peaks and uses the space between two suitable peaks as one cycle. It also understands that a sensor may have been mounted in the opposite direction, so the important peaks may point upward or downward.

A cycle is accepted only when:

- it lasts between 0.60 and 2.00 seconds;
- it contains real, complete measurements;
- it has visible movement;
- it follows the peak rules.

The program needs at least ten good cycles from the left leg and ten from the right leg. If it cannot find enough, it explains why the participant was excluded.

## Step 5: Let a person check the detected cycles

The computer is not allowed to say, “I found cycles, so I must be correct.”

It creates a picture for each participant showing:

- the cleaned left-leg signal;
- the cleaned right-leg signal;
- where each accepted cycle begins.

It also creates a checklist table. The table warns us when, for example:

- there are not many cycles;
- one leg has many more cycles than the other;
- the walking rhythm changes a lot;
- many cycles sit very close to the allowed time limits;
- a large part of the recording is missing.

A warning does not automatically remove anyone. It simply tells the researcher, “Please look at this participant carefully.”

## Step 6: Describe each leg

The program summarizes the accepted cycles from each leg.

It asks:

- How long does a normal cycle take?
- How much does the leg rotate?
- Are the cycle times steady or irregular?
- Are the movement sizes steady or irregular?
- What does the usual movement shape look like?

It uses the middle, or median, cycle values so one strange step cannot easily control the result.

## Step 7: Compare the two legs in four ways

The program makes four comparisons.

### 1. Timing difference

Do the left and right legs take similar amounts of time?

### 2. Movement-size difference

Do the two legs rotate through a similar movement size?

### 3. Steadiness difference

Is one leg much more irregular than the other?

### 4. Movement-shape difference

Do the two legs make a similarly shaped movement during each cycle?

The program uses the size of the difference, not which side is larger. Therefore, it does not need to know whether the left or right leg was affected.

## Step 8: Learn what “normal difference” looks like

Even healthy people do not walk with perfectly identical legs.

The program first looks at the healthy group and learns the usual amount of difference for each of the four comparisons.

Think of it as learning how much wobble is normal before deciding that a wobble is unusually large.

For each participant, it then asks:

> How far above the usual healthy value is this difference?

A difference below the healthy middle does not add severity.

## Step 9: Avoid the old score ceiling

The older version had a problem. Very large differences were all stopped at the same maximum value.

Imagine measuring people's heights with a ruler that stops at 180 cm. Someone who is 181 cm and someone who is 200 cm would both be written down as 180 cm. We would lose the difference between them.

This happened with the old waveform score: all stroke participants reached the same ceiling.

The new program uses a soft curve instead. Large values are gently squeezed so they cannot overpower everything else, but they are never made completely equal. A larger difference still receives a larger value.

## Step 10: Make one overall score

The program combines the four comparisons:

- timing;
- movement size;
- steadiness;
- movement shape.

Each comparison receives one-quarter of the final score.

A smaller score means the legs behave more similarly. A larger score means the sensors found more bilateral gait asymmetry.

## Step 11: Create the healthy class

All valid healthy participants are placed in the Low/reference class.

The program still checks the range of healthy scores. This helps us understand when a stroke participant looks similar to healthy participants. However, the healthy boundary no longer decides whether a known stroke participant belongs to a stroke class.

This matters because we already know which people are stroke participants. The question inside the stroke group is whether their sensor-derived asymmetry is lower or higher relative to the other stroke participants.

## Step 12: Divide the stroke participants into two groups

The program lines up all stroke participants from the smallest score to the largest score.

It then makes a cut close to the middle:

- the lower-scoring stroke participants enter the Moderate group;
- the higher-scoring stroke participants enter the High group.

The cut is placed between two actual scores, not through one participant's score. The program also requires at least five people in each stroke group.

With 15 valid stroke participants, this normally gives:

- 7 participants in one stroke group;
- 8 participants in the other stroke group.

This is a fair and usable split for the current small dataset. It does not mean nature has proven that there are exactly two medical types of stroke severity.

## Step 13: Check whether the middle line is stable

Imagine putting all the stroke names into a bag, drawing names many times, and calculating the middle line again.

The program performs a computer version of this experiment 500 times. This is called bootstrapping.

If a participant stays on the same side almost every time, the assignment is stable. If the participant often changes sides, the program marks the assignment as borderline.

A borderline participant is not deleted. The mark simply tells us that the person is close to the boundary and that we should be careful when interpreting the class.

## Step 14: Freeze the recipe

After the score and class rule are approved, the program writes the complete recipe into `frozen_score_definition.json`.

It records:

- which four measurements were used;
- how healthy values were calculated;
- how large values were softened;
- how the four measurements were combined;
- where the stroke-group boundary was placed;
- which version of the recipe was used.

It also creates a digital fingerprint. If someone changes the recipe later, the fingerprint changes too.

This is like sealing the recipe in an envelope before testing the machine-learning models. We must not keep changing the recipe just because one change gives better classification accuracy.

## Step 15: Do not give the classifier the answer

The classes are created using gyroscope-z measurements.

If we give those exact same measurements to the main classifier, it may simply rebuild the formula instead of learning an independent relationship.

That would be like giving a student the answer sheet during an exam.

Therefore, the main classifier should use other information, such as:

- accelerometer features;
- gyroscope x/y features;
- available EMG features.

The exact gyroscope-z measurements used to create the labels should be excluded from the primary experiment.

## What the three classes really mean

The classes mean:

- **Low/reference:** a participant from the healthy reference group;
- **Moderate:** a stroke participant in the lower part of the study's sensor-derived asymmetry ordering;
- **High:** a stroke participant in the higher part of that ordering.

They do not mean:

- medically mild, moderate, or severe stroke;
- a doctor's diagnosis;
- a proven rehabilitation-outcome level;
- a permanent threshold that automatically applies to every hospital and population.

The safest description is:

> Relative sensor-derived gait-asymmetry groups within this study sample.

## The files produced at the end

The program creates:

- one table with the final participant scores and classes;
- one table containing every accepted gait cycle;
- one table describing the healthy reference values;
- one table listing cycle-quality warnings;
- one image for checking cycle detection for each participant;
- one file containing the frozen score recipe;
- one summary containing settings, thresholds, class counts, and warnings;
- one exclusion table when a participant cannot be processed;
- synchronized sensor files when saving them is enabled.

## The whole process in one short story

The program makes the sensor clocks agree, cleans the signals, finds repeated walking cycles, checks whether those cycles look believable, compares the two legs in four ways, learns how much difference is normal in healthy people, and combines the four comparisons into one score.

Healthy participants form the reference class. Stroke participants are lined up by score and split near the middle into lower- and higher-asymmetry groups, with enough people required in both groups. The computer repeats the split many times to show which assignments are stable and which are close to the boundary. Finally, it seals the scoring recipe so it cannot be changed after seeing the machine-learning results.
