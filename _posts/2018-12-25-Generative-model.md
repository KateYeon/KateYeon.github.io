---
title: Generative Models
categories:
- Business Analytics
excerpt: |
 discriminative model과 generative model의 차이점과 MLE와 GMM를 사용한 case study를 소개하겠습니다. 이를 바탕으로 정리된 generative model python code를 설명과 함께 첨부합니다.
feature_text: |
  ## Generative Models
  이 포스팅은 고려대학교 강필성 교수님의 비즈니스애널러틱스 강의자료와 강성호 학생분의 코딩을 바탕으로 작성되었습니다.
feature_image: "https://picsum.photos/2560/600?image=872"
image: "https://picsum.photos/2560/600?image=733"
---

### Discriminative vs Generative

discriminative 모델은 우리가 흔히 사용하는 다중회귀식을 생각해 볼수 있다. 주어진 Input data를 사용해서 y를 어디에 할당할 것인가?의 문제인 것이다.
generative 모델은 class할당의 logic에서 joint distribution을 적용한다. 먼저 class에 대한 확률분포를 정의하고 p(y) 그 클래스로부터 x라는 설명변수가 나왔다고 정의한다.discriminative model은 설명변수만 주어지면 어떤분포로부터 설명변수가 생성되었는지는 관심 없고 생성된 class변수를 바탕으로 학습시켜서 후에 class예측 또는 분류만을 목표로 했다면, generative model은 데이터가 어떤 메커니즘에 의해서 생성되었는지 판별,설명 또는 설명변수 분포를 생성하고 싶어한다. 따라서, 설명변수와 종속변수의 조합은 어떻게 생성되었는지에 대한 생성확률을 최대화 하는 모델을 학습시키고자한다. 

예를 들어서, discriminantive model 와 generative model 모두 분류가 모델의 목적일때, 두 모델은 decision boundary를 추정한다.  discriminantive model 은 주어진 설명변수와 labled종속변수 정보만으로 모델을 학습시키고, 분류경계면을 추정하여 미래의 input data에 대한 분류를 한다.

반면, genrative model의 관점은 data가 modal이 2개인 가우시안 분포로부터 파생되었다는 가정을 하여, 처음에는 labeled data를 사용해서, 초기 mean과 covariance matrix를 구하고, unlabeled data의 각각의 점에서 해당하는 modal에서부터의 확률분포를 추정한다. 어느지점부터는 각 modal으로 부터 생성확률이 같아지는 부분이 있을 것이다. 그 점로 이루어진 경계면으로, decision boundary를 추정한다. 따라서, unlabeled data를 포함하여 분류경계면을 추정하면서, 생성데이터의 확률분포를 가정하에 추정해볼 수 있다는 장점이 있다.

즉, 각각의 class가 어떻게 생성되었는지를 확률분포를 고려해서 모델학습 과정에 반영하면 generative model인 것이고, class가 어떻게 생성되었는지 확률분포를 고려하지 않은채, labeled 결과만을 모델학습 과정에 반영하면 discriminative model인 것이다. 

만약,데이터에 class 불균형이 존재한다면 prior probablity를 고려해서 더 높은 쪽으로 할당하여 분류해 볼 수 있다.

### A simple example of generative models


지금까지는 discriminantive model과 generative model을 서로 설명, 비교하며 generative model의 대략적인 이해를 도왔다. 이번에는 간단한 simulation 그림으로 이해를 해보자.

![Imgur](https://i.imgur.com/wROlBf6.png?1)

위의 그림은 두 class 를 갖는 labeled simulation data (X,Y)를 그린 그림이다. 만약 위의 simulation data에서 class 각각 gausian distribution에서 생성되었다는 가정을 한다면 Generative model을 사용하여 두 class의 분류경계면은 어떻게 추정될 수 있을까?

먼저 다음과 같은 가정을 한다.

![Imgur](https://i.imgur.com/8aOqZuF.png)

data의 생성 분포를 modal=2의 가우시안분포를 가정했다. parameter setting을 살펴보면, 각 클래스의 비중과 평균 covariance matrix를 전부 구함으로써 생성분포의 확률분포를 각각 전부 구하려는 것을 알 수 있다.

다음으로는 GMM식을 살펴보자. 왼쪽항은 각 범주에 대한 mean vector, covariance matrix,가중치를 안다는 가정하에서, x와 y에 상응하는 label이 얼만큼의 생성확률을 갖는지 물어보는 것이다. 오른쪽항은 두 파트로 구분된다. 첫번째 파트는 hyperparameter상에서 y라는 class가 나올 확률이고, 두번째 파트는 y가 고정되었을때 x가 나올 확률을 의미한다. 

두파트의 곱은, 가중치와 설정한 분포의 곱으로 표현된다. 이때, 가중치는 빈도를 의미하여 해당하는 y범주에 대한 빈도를 사용한다. 가우시안분포로 가정했기 때문에, 해당하는 mean과 covaraince를 갖는 가우시안 분포로 정리할 수 있다.

여기까지 정리한 수의 분류는, 베이지안 정리를 사용하여 x가 주어졌을때 y(class)를 할당할 확률식을 표현할 수 있다.

labled data만을 사용하여 각각 mle로 추정한 모수들을 갖는 가우시안분포를 그려본 모습이다.

![Imgur](https://i.imgur.com/pjZJevv.png)

초록색선이 두 class분포로부터 같은 확률을 갖는 점들을 이은 경계면이며 이를 decision boundary로 사용한다.

![Imgur](https://i.imgur.com/24GS4xX.png)

여기에서 unlabled data를 추가해보자. 흩뿌려진 초록색점들의 unlabled data를 포함하여 분류경계면추정을 한다면 어떻게 decision boundary를 그린 모습은 아래와 같다.

![Imgur](https://i.imgur.com/C8q4Hm9.png)

labeled data만을 사용하여 class boundary를 그린모습과, unlabled data(초록색 점들)를 포함하여 boundary를 그린 모습을 비교해보면, 각각의 class 생성분포또한 분명하게 변했으며, 분류경계면 또한 확실히 변해있는 것을 확인 할 수 있다.

![Imgur](https://i.imgur.com/Np7gym3.png)

이 두 경계면이 다른 이유는 사실은 당연하다.
둘은 서로 다른 생성확률분포를 maximize하기 때문이다.




## Case study: GMM


### Generative model for semi-supervised learning

![Imgur](https://i.imgur.com/2D5elUj.png)

우리가 관심있는 함수식은 다음과 같다. 우리는 Yu의 조합을 알지 못하기 때문에 Yu의 모든 경우의 수를 전부다 고려하여 더해줌으로써, interested qauntity를 구할 수 있다. 모수를 추정하는 방법은 대표적으로 MLE가 있으며, MAP 또는 베이지안 방법을 적용시킬수 있다. 설명의 단순 명료함을 위해여, MLE 추정 모수를 바탕으로한 GMM을 적용하여 이항분류를 해보자.

첫번째로, 아래의 식으로 labeled data만을 사용하여 mle를 구한다.

![Imgur](https://i.imgur.com/jRYylWn.png)

이때의 MLE는 이미 정리된 labled data의 frequency statistics이므로 매우 trivial하다.

두번째로, labled data에 unlabled data를 추가하여 아래의 object function을 정리하자.

![Imgur](https://i.imgur.com/GrhOupr.png)

이와 같이 구하는 이유는 unlabled Y는 hidden variable이므로 명시적으로 mle를 구하기가 어렵기 때문이다. 왼쪽항의 첫번째식은 위에서 구한 labled data만을 사용한 식과 동일하다. 빨간색으로 구별한 두번째 식이 추가된 것이다. y를 이항분류로 설정하였으므로, y는 1또는 2일 수 있다. 이때 1범주인지 2범주인지모르므로 모든경우의 수를 더 해준다. l+1 부터 l+u까지의 u개의 unlabled data를 추가한다.

다음으로는 정리된 object function의 local optimum을 찾기 위하여 유용하게 널리 사용되는  Expectiation-Maximization(EM) algorithm을 적용하여 최종적으로 문제를 풀어보자.

### The EM algorithm for GMM

이 단계에서는 0단계와 1단계(E-step) 2단계(M-step)로 설명할 수있고, maximum iteration까지 1,2단계를 반복해서 문제를 풀 수 있다.

![Imgur](https://i.imgur.com/vAb77z8.png)

그림으로 설명을 step1 의 덧붙여 설명해본다면,아래와 같다. 
unlabeld data(초록색점들)을 빨간색 X표자로 대신해서 설명하면, 각각의 빨간색X점에 대하여 class(Y)가1일때의 선택한 x가 생성될 확률과 class(Y)가 2일때의 선택한 x가 생성될 확률을 구할 수 있다.

![Imgur](https://i.imgur.com/DHciDC4.png)

결국 키는 처음에 설정한 object function을 maximize 하는 것이다.
현재의 case study에서는 EM algorithm을 사용하였지만, 이는 object function을 maximize하기위한 하나의 방법일 뿐이다. 다른 방법들도 충분히 가능하다. 다른 방법으로는 vriational approximation 또는 direct optimization방법이 있다.

### Heuristics to lessen the danger

generative model의 위험요소를 조심스럽게 다루기 위하여, unlabled data앞에 down weight(1보다 작은)를 곱해주는 방법이 있다.

![Imgur](https://i.imgur.com/o5N3zKK.png)


## Phython code


![Imgur](https://i.imgur.com/lLipvIg.png)
![Imgur](https://i.imgur.com/G7G2arq.png)
![Imgur](https://i.imgur.com/G3JTr8Y.png)
![Imgur](https://i.imgur.com/p8J2L9V.png)
![Imgur](https://i.imgur.com/5zppOET.png)
![Imgur](https://i.imgur.com/6z26UJc.png)
![Imgur](https://i.imgur.com/O0zgl9A.png)
![Imgur](https://i.imgur.com/MwyKMOo.png)
![Imgur](https://i.imgur.com/61bfLIQ.png)
![Imgur](https://i.imgur.com/6sQtZpq.png)
![Imgur](https://i.imgur.com/qzGxJvJ.png)


Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

#### reference
강필성(비즈니스 애널러틱스 강의자료,2018) ,Xiaojin Zhu(2007), choi(2015), 강성호(generative model code,2017)


[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
