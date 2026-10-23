import assert from 'node:assert/strict';import{checkReview,factStrings,reviewPrompt}from'../src/review-check.ts';
const g={pass:false,required:0.005,current:{brier:0.23456,accuracy:0.61},candidate:{brier:0.2331,accuracy:0.6},matches:182},f=factStrings(g);
assert.match(reviewPrompt(g),/VERDICT: KEEP/);
assert.equal(checkReview(`VERDICT: KEEP\nChallenger Brier ${f.challengerBrier} vs ${f.currentBrier}: improvement ${f.improvement} is below ${f.required}; accuracy ${f.challengerAccuracy} < ${f.currentAccuracy}.`,g).ok,true);
assert.equal(checkReview('VERDICT: PROMOTE\nShip it.',g).verdictOk,false,'contradicting verdict rejected');
assert.equal(checkReview('No verdict line.',g).verdictOk,false,'missing verdict rejected');
assert.deepEqual(checkReview('VERDICT: KEEP\nAccuracy would reach 75.0%.',g).unknown,['75.0%'],'invented number flagged');
console.log('review-check: verdict agreement and invented-number detection pass');
