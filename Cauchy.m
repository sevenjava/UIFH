function Cau = Cauchy(X, u0, v0)
% the bivariate Cauchy distribution function

  [h, w] = size(X);
  d = max(h, w);
  c = ceil(d/4);
  Cau = single(zeros(h, w));
  for x = 1:h
      for y = 1:w
         Cau(x,y) = (1/pi) * c / ((((x-u0).^2+(y-v0).^2)+c.^2).^(3/2));       
      end
  end
end