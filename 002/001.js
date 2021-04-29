let grade_v = [4, 1, 0, 3, 4, 1, 0, 1, 0, 2];

let dcg = function (index, list) {
  let rel = 0;
  let res = 0;
  for (let i = 0; i < index; i++) {
    if (list[i] >= 0) {
      rel++;
    }
    res += list[i] / Math.log2(i + 2);
    console.log(list[i] / Math.log2(i + 2));
  }

  return res;
};

let idcg = function (index) {
  let grade_v_sorted = grade_v.sort((a, b) => b - a);
  return dcg(index, grade_v_sorted);
};

console.log(dcg(5, grade_v) / idcg(5, grade_v));
