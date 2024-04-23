function t = nu2stab(x)
%NUM2STR Number to string conversion.
%	T = NUM2STR(X)  converts the scalar number  X into a string
%	representation  T  with about  4  digits and an exponent if
%	required.   This is useful for labeling plots with the
%	TITLE, XLABEL, YLABEL, and TEXT commands.
%
%	See also INT2STR, SPRINTF, FPRINTF.

%	Copyright (c) 1984-93 by The MathWorks, Inc.

%	Modification 1993 par Gautier/LAN Robotique
%	pour affichage de resultats dans tableaux
%	See also NUM2STE, NUM2STF

if isstr(x)
    t = x;
else
    t = sprintf('%12.4f',real(x));
    if imag(x) > 0
       t = [t '+' sprintf('%12.4f',imag(x)) 'i'];
    elseif imag(x) < 0
       t = [t '-' sprintf('%12.4f',imag(x)) 'i'];
    end
end
