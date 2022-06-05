# Re-evaluation of submissions to the BOP Challenge 2019

**On 27th January 2020, all submissions to the BOP Challenge 2019 were re-evaluated after discovering and fixing two bugs in the evaluation system. The change in scores is not dramatic and the ordering of the participated methods stays largely the same. The winners of [all presented awards](https://bop.felk.cvut.cz/media/bop_challenge_2019_results.pdf) stay the same. We are sorry for any inconvenience caused.**

Scores before the re-evaluation can be found [here](https://docs.google.com/spreadsheets/d/1EHxOsktqPKCZWwmSTPj7CDHxkhHP6YIx_X083zMMhws/edit?usp=sharing). Scores after the re-evaluation are visible in the leaderboards on [bop.felk.cvut.cz](https://bop.felk.cvut.cz/leaderboards/bop19_core-datasets/).
[Slides presenting the challenge winners](https://bop.felk.cvut.cz/media/bop_challenge_2019_results.pdf) and [slides from the R6D'19 workshop](http://cmp.felk.cvut.cz/sixd/workshop_2019/slides/r6d19_hodan_bop_challenge_2019.pdf) were updated with the new scores.

## Bug fixes

[Bug fix #1](https://github.com/thodan/bop_toolkit/commit/48bc1ede8b97fbcd5e3fe67d23b9a1a31b48fe73#diff-a487982cee6cf560d39006ebf957f1f0) - Casting the discretization step for continuous rotational symmetries to *bool* instead of *float* caused that the continuous rotational symmetries were not considered in the evaluation. Thanks to github user *georgegu1997* for [pointing out](https://github.com/thodan/bop_toolkit/issues/27) this issue.

[Bug fix #2](https://github.com/thodan/bop_toolkit/commit/975fcf25cb77529a92176cbffc4a4b87ad6d0e20#diff-d6f3948fed929e1bb7e868ade3afc7bc) - The participants were asked to provide *k* pose estimates per object model and image. The value of *k* was defined as the number of instances which are visible from at least 10% and was provided through the *inst_count* item in *test_targets_bop19.json*. For some datasets, *inst_count* was calculated using the visibility test described in Hodan et al. (ECCVW'16) and for some using the modified visibility test described [here](https://bop.felk.cvut.cz/challenges/bop-challenge-2019/#evaluationmethodology) (in the paragraph about VSD). The two visibility tests may yield slightly different values of *inst_count*. The evaluation script was internally using the modified visibility test for all datasets. After the fix, the evaluation script reads the value *inst_count* from *test_targets_bop19.json* and thus always assumes the number of estimates which the participants were asked to provide.

## Test data

* [gt](https://bop.felk.cvut.cz/media/data/bop_sample_results/bop_challenge_2019/gt/) - GT poses.
* [gt-equivalent](https://bop.felk.cvut.cz/media/data/bop_sample_results/bop_challenge_2019/gt-equivalent/) - Each GT pose was replaced by a random pose from the "equivalence class" given by the [object symmetries](https://github.com/thodan/bop_toolkit/blob/master/scripts/vis_object_symmetries.py).

Scores achieved with *gt* and *gt-equivalent* test data can be found [here](https://docs.google.com/spreadsheets/d/1dK4OYUpAqKYUpc-by-XqnX3F5NqcbqekDlGnDi5h0G4/edit?usp=sharing). Test data *gt* achieves 100% on all scores, *gt-equivalent* achieves 100% on AR_MSSD and AR_MSPD, and 98.9+% on AR_VSD (the threshold setting for VSD is a bit stricter and the surface of the object model in two "symmetrical" poses is not always perfectly aligned).