# 004 · Forecasting spare-parts demand that is mostly zeros

## Problem
A maintenance storeroom has to decide how many of each spare part to keep. Most parts aren't used at all in a typical week, then someone needs three at once. Normal forecasting methods handle this kind of "intermittent" demand badly. And the usual accuracy score can point you to a forecast that would leave the shelf empty.

## Inputs
104 weeks of demand for 60 spare parts. **All synthetic**, generated in `forecast.py`. Each part has its own chance of being needed in a given week (5-40%) and its own typical quantity when it is (1-6 units). 75% of part-weeks have zero demand. The first 78 weeks are for training and the last 26 weeks are the test.

## Approach
1. **Five forecasts**, each made one week ahead and updated as the actual demand comes in:

| Forecast | Idea |
|---|---|
| Zero | Always predict 0. Only here to show a scoring trap |
| Fixed average | Training-period average, never updated (like a min/max set once) |
| Moving average | Mean of the last 12 weeks |
| Exponential smoothing (SES) | Move a little toward every new week, zeros included |
| Croston (SBA) | Track *how much* when it happens and *how often* it happens separately. Only update when there is demand |

2. **Same storeroom rule for all.** Every week, top the stock up to: forecast × 3 weeks (2-week lead time + 1 week until the next review) + safety stock. Safety stock = z × each method's own forecast error on the training weeks. Demand the shelf can't cover is a shortfall.
3. **Two ways to compare.** (a) Use z = 1.65, set in advance, and look at fill rate and stock. (b) Raise z until each method serves at least 95% of units, then compare how much stock that took. (b) is the fair comparison, because higher stock in (a) also buys more fill.

## Results
Over the 26 test weeks (1,346 units demanded):

| Forecast | MAE | Bias | Fill rate (z=1.65) | Units on shelf (z=1.65) | Units on shelf for 95% fill |
|---|---|---|---|---|---|
| Zero | **0.86** | -100% | – | – | – |
| Fixed average | 1.27 | -0.6% | 96.4% | 362 | **315** |
| Moving average (12w) | 1.29 | +2.0% | 97.8% | 416 | 345 |
| Exponential smoothing | 1.29 | +2.6% | 98.1% | 412 | 337 |
| Croston (SBA) | 1.27 | -2.7% | 97.0% | 369 | 328 |

## Takeaway
**MAE picked the worst forecast.** When three-quarters of weeks are zero, predicting zero gets the best MAE score, even though it would never order anything. For intermittent demand, look at bias and at what the forecast does to stock, not at MAE.

**Calm forecasts need less stock.** Moving average and SES jump up right after a big order and then slowly come back down. So the storeroom restocks just after demand happened, when it's least likely to be needed again. For the same 95% fill, they needed 3-10% more units on the shelf than Croston or the fixed average. On five other seeds, Croston or the fixed average also needed the least stock every time.

**A fixed average was as good as Croston here.** That's because the synthetic demand doesn't change over time. Croston's value is that it keeps learning without chasing spikes, which only pays off when demand does shift (new equipment, aging machines). Also note that z = 1.65 delivered 96-98% fill, not 95%: the errors aren't normally distributed, so check the safety factor against actual fill instead of trusting the textbook value.

## Next
- Add parts whose demand trends up or steps up, and see if Croston beats the fixed average there
- Classify parts first (how often and how variable their demand is) and use a different method per class

## Run
```bash
pip install -r ../../requirements.txt
python forecast.py
python -m pytest
```
