package com.example.demo.Mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.example.demo.Entity.tweet;
import com.example.demo.Entity.tweetETO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;
import java.util.Map;

@Mapper
public interface tweetMapper extends BaseMapper<tweet> {
    @Select("${TweetPositive}")
    String SqlString(@Param("TweetPositive")String sql);

    @Select("${findtext}")
    List<tweet> FindText(@Param("findtext")String sql);



}
