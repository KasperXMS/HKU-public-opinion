package com.example.demo.Mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.example.demo.Entity.tweetDto;
import com.example.demo.Entity.weiboDto;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface weiboDtoMapper extends BaseMapper<weiboDto> {
    @Select("${findtext}")
    List<weiboDto> FindText(@Param("findtext")String sql);
}
