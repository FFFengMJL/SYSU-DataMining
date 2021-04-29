const loopTime = 100;
const dotNumList = [5, 10, 20, 30, 40, 50, 60, 70, 80, 100];

/**
 * 随机获取一个点，判断是否在函数积分中
 * @param {Function} func 需要计算定积分的函数
 * @returns {Boolean} 如果这个点在积分中，返回 true；否则返回 false
 */
function getOneDot(func) {
  let x = Math.random();
  let y = Math.random();

  return y <= func(x);
}

/**
 * 根据点的数量计算积分
 * @param {Number} dotNum
 * @param {Function} func
 * @returns {Number} 返回蒙特卡洛方法采样得到的积分的近似值
 */
function getAMap(dotNum, func) {
  let res = 0;
  for (let i = 0; i < dotNum; i++) {
    if (getOneDot(func)) {
      res++;
    }
  }

  return res / dotNum;
}

/**
 * 需要计算的定积分函数
 * @param {Number} x 横坐标
 * @returns {Number} 纵坐标
 */
function originFunction(x) {
  return Math.pow(x, 3);
}

function main() {
  let resList = [];
  for (let num of dotNumList) {
    resList.push([]);
    let mean = 0; // 均值
    let variance = 0; // 方差

    for (let i = 0; i < loopTime; i++) {
      let intergration = getAMap(num, originFunction); // 当前结果
      resList[resList.length - 1].push(intergration);
      mean += intergration;
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
