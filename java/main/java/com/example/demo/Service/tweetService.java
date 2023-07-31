package com.example.demo.Service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.example.demo.Entity.tweet;

import java.util.List;
import java.util.Map;

public interface tweetService extends IService<tweet> {

    String countnumber();
    String AvgValue();
    Map DateVal();
    Map Hotval();
    List FindPositiveText();
    List FindNegativeText();
    List FindText();
    List FindWordCloud();
}
