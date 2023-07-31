package com.example.demo.Service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.example.demo.Entity.weibo;

import java.util.List;
import java.util.Map;

public interface weiboService extends IService<weibo> {
    String countnumber();
    String AvgValue();
    Map findmap();
    Map DateVal();
    Map HotVal();
    List FindPositiveText();
    List FindNegativeText();
    List FindText();
    List FindWordCloud();
    List FindOverviewCloud();
}
