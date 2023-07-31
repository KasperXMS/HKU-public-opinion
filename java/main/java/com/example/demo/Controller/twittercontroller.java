package com.example.demo.Controller;

import com.example.demo.Entity.tweet;
import com.example.demo.Mapper.tweetMapper;

import com.example.demo.Service.tweetService;
import com.example.demo.common.returnvalue;
import javafx.beans.binding.ObjectExpression;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/tweet")
public class twittercontroller {


    @Autowired
    private tweetService tweetService;

    @CrossOrigin
    @GetMapping
    public returnvalue<Map> findText(){
        Map<Object,Object> map=new HashMap<>();
        Map<Object,Object> TweetPositiveText=new HashMap<>();
        Map<Object,Object> TweetNegativeText=new HashMap<>();
        Map<Object,Object> TweetText=new HashMap<>();
        TweetText.put("TweetText",tweetService.FindText());

        TweetPositiveText.put("TweetPositiveText",tweetService.FindPositiveText());
        TweetNegativeText.put("TweetNegativeText",tweetService.FindNegativeText());
        map.put("TweetText",TweetText);
        map.put("TweetPositiveText",TweetPositiveText);
        map.put("TweetNegativeText",TweetNegativeText);
        map.put("TweetCloud",tweetService.FindWordCloud());
        return returnvalue.success(map);
    }

}
