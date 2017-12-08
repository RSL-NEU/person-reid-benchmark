%%
function p_a = p_lab(x)
    p_a=[];
    if x >= -110 && x <=-55
        p_a = 1;
    end
    
    if x > -55 && x <=0
        p_a = 2;
    end
    
    if x > 0 && x <=55
        p_a = 3;
    end
    
    if x > 55 && x <=110
        p_a = 4;
    end
end