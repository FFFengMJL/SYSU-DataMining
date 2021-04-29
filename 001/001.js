const loopTime = 20;
const dotNums = 20; // 投点个数分别是20, 50, 100, 200, 300, 500, 1000, 5000时，pi值分别是多少
const dotNumsList = [20, 50, 100, 200, 300, 500, 1000, 5000];

/**
 * 随机获取一个点，并判断是否在 1/4 个圆上
 * @returns 返回这个点是否在圆上；如果在，返回 true，否则返回 false
 */
function getOneDot() {
  let x = Math.random();
  let y = Math.random();
  return Math.pow(x, 2) + Math.pow(y, 2) <= 1;
}

/**
 * 获取一张图
 * @param {Number} dotNums 点的个数
 * @returns 蒙特卡洛方法计算出的 pi 值
 */
function getAMap(dotNums) {
  let circle = 0;
  for (let i = 0; i < dotNums; i++) {
    if (getOneDot()) {
      circle++;
    }
  }

  return 4 * (circle / dotNums);
}

function main() {
  let resList = [];
  for (let num of dotNumsList) {
    resList.push([]);
    let mean = 0; // 均值
    let variance = 0; // 方差

    for (let i = 0; i < loopTime; i++) {
      let pi = getAMap(num); // 当前结果
      resList[resList.length - 1].push(pi);
      mean += pi;
    }
    mean /= loopTime; // 计算均值

    // 计算方差
    for (let res of resList[resList.length - 1]) {
      variance += Math.pow(mean - res, 2);
    }
    variance /= loopTime;

    console.log(
      `点数：${num} \t 均值：${mean.toFixed(4)} \t 方差：${variance.toFixed(8)}`
    );
  }
}

main();
