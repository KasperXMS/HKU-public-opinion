package com.example.demo.Mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.example.demo.Entity.tweet;
import com.example.demo.Entity.tweetDto;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface tweetDtoMapper extends BaseMapper<tweetDto> {
    @Select("${findtext}")
    List<tweetDto> FindText(@Param("findtext")String sql);
}
